/** 
 * File:    short_term_tracker.cpp 
 * 
 * Summary of File: Short term people tracker.          
 */ 

#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <unistd.h>
#include <sys/stat.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d.hpp>
#include <deque>
#include <list>

#include <stdio.h>
#include <ctime>

#include <torch/script.h> 

using namespace std;

torch::jit::script::Module feature_module;

void head_feature(const cv::Mat &im, vector<float> &v)
{

           cv::Mat image;
           cv::Mat im2;
           cv::resize(im, im2, cv::Size(128, 128));
           im2.convertTo(image, CV_32FC3, 1.0/255.0, 0);

           cv::Mat bgr[3]; 
           cv::split(image, bgr);
           cv::Mat channelsConcatenated;
           vconcat(bgr[0], bgr[1], channelsConcatenated);
           vconcat(channelsConcatenated, bgr[2], channelsConcatenated);

           cv::Mat channelsConcatenatedFloat;
           channelsConcatenated.convertTo(channelsConcatenatedFloat, CV_32FC3);

           vector<int64_t> sizes = {1, 3, 128, 128};
           at::TensorOptions options(at::ScalarType::Float);
           at::Tensor tensor_image = torch::from_blob(channelsConcatenatedFloat.data, at::IntList(sizes), options);
           vector<torch::jit::IValue> inputs;
           inputs.push_back(tensor_image.to(at::kCUDA));
           auto res = feature_module.forward(inputs);
           at::Tensor result = res.toTensor();
           auto r = result.to(at::kCPU);
           auto out = r.accessor<float, 2>();
           for (int i = 0; i < 128; ++i)
           {
                   v.push_back(out[0][i]);
           }

}

struct Trajectory {
     deque<vector<float> > Q;
     int life;
     int age;
     int id;
     vector<float> appearance;

     Trajectory()
     {
         life = 5;
         age = 0;
         id = -1;
     }

     Trajectory(float x1, float y1, float x2, float y2, float color, float strength, float tid)
     {
         Q.push_back({x1, y1, x2, y2, color, strength});
         life = 5;
         age = 0;
         id = tid;
     }

     void extend(float x1, float y1, float x2, float y2, float color, float strength)
     {
         Q.push_back({x1, y1, x2, y2, color, strength});
         ++age;
         if (Q.size() > 10)
            Q.pop_front();
         life = 5;
     }

     void mantain(const cv::Mat &ht, const cv::Mat &image)
     {
         vector<float> t = Q.back();
         float h00 = ht.at<double>(0,0);
         float h01 = ht.at<double>(0,1);
         float h02 = ht.at<double>(0,2);
         float h10 = ht.at<double>(1,0);
         float h11 = ht.at<double>(1,1);
         float h12 = ht.at<double>(1,2);
         float h20 = ht.at<double>(2,0);
         float h21 = ht.at<double>(2,1);
         float h22 = ht.at<double>(2,2);

         float xw1 = (h00*(t[0]/6)+h01*(t[1]/6)+h02)/(h20*(t[0]/6)+h21*(t[1]/6)+h22)*6;
         float yw1 = (h10*(t[0]/6)+h11*(t[1]/6)+h12)/(h20*(t[0]/6)+h21*(t[1]/6)+h22)*6;

         float xw2 = (h00*(t[2]/6)+h01*(t[3]/6)+h02)/(h20*(t[2]/6)+h21*(t[3]/6)+h22)*6;
         float yw2 = (h10*(t[2]/6)+h11*(t[3]/6)+h12)/(h20*(t[2]/6)+h21*(t[3]/6)+h22)*6;
	 
         if (xw1 < 0 || xw2 >= image.cols || yw1 < 0 || yw2 >= image.rows)
	 {
		 Q.push_back({xw1, yw1, xw2, yw2, t[4], t[5]});
		 life = -1;
	 }
	 else
	 {
                 Q.push_back({xw1, yw1, xw2, yw2, t[4], t[5]});
                 --life;
         }
     }
     
};

/**
 * mincostmatching is from https://github.com/jaehyunp/stanfordacm/blob/master/code/MinCostMatching.cc
 */

typedef vector<double> VD;
typedef vector<VD> VVD;
typedef vector<int> VI;

double mincostmatching(const VVD &cost, VI &Lmate, VI &Rmate) {
  int n = int(cost.size());

  // construct dual feasible solution
  VD u(n);
  VD v(n);
  for (int i = 0; i < n; i++) {
    u[i] = cost[i][0];
    for (int j = 1; j < n; j++) u[i] = min(u[i], cost[i][j]);
  }
  for (int j = 0; j < n; j++) {
    v[j] = cost[0][j] - u[0];
    for (int i = 1; i < n; i++) v[j] = min(v[j], cost[i][j] - u[i]);
  }

  // construct primal solution satisfying complementary slackness
  Lmate = VI(n, -1);
  Rmate = VI(n, -1);
  int mated = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (Rmate[j] != -1) continue;
      if (fabs(cost[i][j] - u[i] - v[j]) < 1e-10) {
        Lmate[i] = j;
        Rmate[j] = i;
        mated++;
        break;
      }
    }
  }

  VD dist(n);
  VI dad(n);
  VI seen(n);

  // repeat until primal solution is feasible
  while (mated < n) {

    // find an unmatched left node
    int s = 0;
    while (Lmate[s] != -1) s++;

    // initialize Dijkstra
    fill(dad.begin(), dad.end(), -1);
    fill(seen.begin(), seen.end(), 0);
    for (int k = 0; k < n; k++)
      dist[k] = cost[s][k] - u[s] - v[k];

    int j = 0;
    while (true) {

      // find closest
      j = -1;
      for (int k = 0; k < n; k++) {
        if (seen[k]) continue;
        if (j == -1 || dist[k] < dist[j]) j = k;
      }
      seen[j] = 1;

      // termination condition
      if (Rmate[j] == -1) break;

      // relax neighbors
      const int i = Rmate[j];
      for (int k = 0; k < n; k++) {
        if (seen[k]) continue;
        const double new_dist = dist[j] + cost[i][k] - u[i] - v[k];
        if (dist[k] > new_dist) {
          dist[k] = new_dist;
          dad[k] = j;
        }
      }
    }

    // update dual variables
    for (int k = 0; k < n; k++) {
      if (k == j || !seen[k]) continue;
      const int i = Rmate[k];
      v[k] += dist[k] - dist[j];
      u[i] -= dist[k] - dist[j];
    }
    u[s] += dist[j];

    // augment along path
    while (dad[j] >= 0) {
      const int d = dad[j];
      Rmate[j] = Rmate[d];
      Lmate[Rmate[j]] = j;
      j = d;
    }
    Rmate[j] = s;
    Lmate[s] = j;

    mated++;
  }

  double value = 0;
  for (int i = 0; i < n; i++)
    value += cost[i][Lmate[i]];

  return value;
}

float box_dist(float x1, float y1, float x2, float y2, float u1, float v1, float u2, float v2)
{
   float s = sqrt((x1-u1)*(x1-u1) + (y1-v1)*(y1-v1) + (x2-u2)*(x2-u2) + (y2-v2)*(y2-v2)) / (fabs(x2-x1) + fabs(y2-y1));
   return s;
}

float size_dist(float x1, float y1, float x2, float y2, float u1, float v1, float u2, float v2)
{
   return (fabs((x2-x1)/(u2-u1)) + fabs((y2-y1)/(v2-v1)))/2.0;
}

cv::Mat global_motion(cv::Mat &image)
{
   int h = image.rows / 6;
   int w = image.cols / 6;

   static bool first_frame = true;
   
   cv::Mat img = image.clone();
   cv::resize(img, img, cv::Size(w, h));
   cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
   static cv::Mat prev_image;

   if (first_frame)
   {
        prev_image = img.clone();
	first_frame = false;
   }

   cv::Mat flow;
   cv::calcOpticalFlowFarneback(prev_image, img, flow, 0.5, 3, 15, 10, 5, 1.2, 0);

   vector<cv::Point2f> s_points;
   vector<cv::Point2f> t_points; 

   for (int i = 0; i < img.rows; ++i)
   for (int j = 0; j < img.cols; ++j)
   {
	cv::Point2f v = flow.at<cv::Point2f>(i, j);   
       	s_points.push_back(cv::Point2f(j, i));
        t_points.push_back(cv::Point2f(j + v.x, i + v.y));
   }

   cv::Mat H = cv::findHomography(s_points, t_points);

   prev_image = img.clone();

   return H;
}


float appearance_dist(vector<float> a, vector<float> b)
{
    float s = 0;
    for (int i = 0; i < a.size(); ++i)
       s = s + (a[i]-b[i])*(a[i]-b[i]);
    return sqrt(s);
}

int main(int argn, char **argv )
{
     feature_module = torch::jit::load("../../../models/head_feature.pt");
     feature_module.to(at::kCUDA);
	
     vector<vector<int> > colors;
     for (int n = 0; n < 1000; ++n)
     {
         colors.push_back({rand() % 256, rand() % 256, rand() % 256});
     }

     list<Trajectory> traks;
     int tid = 0;

     cv::Mat frame;
     cv::VideoCapture capture(argv[1]);
     if (!capture.isOpened())
     {
         cerr << "ERROR: Can't open " << argv[1] << endl;
         return 1;
     }

     FILE *fp_out = fopen("result.txt", "wt");

     for (;;)
     {
          static int frame_num = 0;
          capture >> frame;

          if (frame.empty()) break;

          auto img = frame.clone();
          cv::resize(img, img, cv::Size(img.cols/2, img.rows/2));
          cv::Mat ht = global_motion(img);

          vector<vector<float> > dets;

          int x1, y1, x2, y2;
          float p_prob;
          char det_fname[1000];
          sprintf(det_fname, "%s/results/%d.txt", argv[2], frame_num);
          if (access( det_fname, F_OK ) == -1) break;
          FILE *fp = fopen(det_fname, "rt");
          while (fscanf(fp, "%d %d %d %d %f", &x1, &y1, &x2, &y2, &p_prob) != EOF)
          {
             //if (p_prob < 0.9) continue;        
             if (x1 < 0) x1 = 0;
             if (y1 < 0) y1 = 0;
             if (x2 >= img.cols) x2 = img.cols-1;
             if (y2 >= img.rows) y2 = img.rows-1;
             if (x1>=x2 || y1>=y2) continue;
             dets.push_back({x1, y1, x2, y2});
          }
          fclose(fp);
	  
          vector<float> vr;
          vector<float> vc;
          vector<float> det_r;
          vector<float> det_c;
          vector<vector<float> > appearance;

          for (int i = 0; i < dets.size(); ++i)
          {
               det_c.push_back((dets[i][0]+dets[i][2])/2);
               det_r.push_back((dets[i][1]+dets[i][3])/2);
	       if (dets[i][0] < 0) dets[i][0] = 0;
	       if (dets[i][1] < 0) dets[i][1] = 0;
               if (dets[i][2] >= img.cols) dets[i][2] = img.cols - 1;
	       if (dets[i][3] >= img.rows) dets[i][3] = img.rows - 1;

               cv::Mat img_cut = img(cv::Rect(dets[i][0], dets[i][1], dets[i][2]-dets[i][0], dets[i][3]-dets[i][1]));
               vector<float> f;
               head_feature(img_cut, f);
               appearance.push_back(f);	       
          } 
    
          static bool first_frame = true;

          for(list<Trajectory>::iterator p = traks.begin(); p != traks.end(); )
          {
             if (p->life < 0)
                p = traks.erase(p);
             else
                ++p;
          }

          if (first_frame)
          {

            for (int i = 0; i < dets.size(); ++i)
            {
               Trajectory x(dets[i][0], dets[i][1], dets[i][2], dets[i][3], 1, 1, tid);
               x.appearance = appearance[i];
               traks.push_back(x);
               ++tid;
            }
            
            first_frame = false;
          }
          else if (dets.size() == 0)
          {  
                 for(list<Trajectory>::iterator p = traks.begin(); p != traks.end(); ++p )
                 {
                       (*p).mantain(ht, img);
                 }               
  
          }
          else if (traks.size() == 0)
          {
              for (int i = 0; i < dets.size(); ++i)
              {
                 Trajectory x(dets[i][0], dets[i][1], dets[i][2], dets[i][3], 1, 1, tid);
                 x.appearance = appearance[i];
                 traks.push_back(x);
                 ++tid;
              }
          }
          else {

               map<int, vector<list<Trajectory>::iterator> > age_map;	       
               for(list<Trajectory>::iterator p = traks.begin(); p != traks.end(); ++p )
               {
                   if (p->age > 10)
                   {
                       if (age_map.find(1000) == age_map.end())
                       {
                           vector<list<Trajectory>::iterator> t;
                           age_map[1000] = t;
                       }

                       age_map[1000].push_back(p);}
                   else
                   {
                       if (age_map.find(100) == age_map.end())
                       {
                           vector<list<Trajectory>::iterator> t;
                           age_map[100] = t;
                       }

                       age_map[100].push_back(p);}
		       
               }

	       vector<int> dets_used(dets.size(), 0);

	       for (auto q = age_map.rbegin(); q != age_map.rend(); ++q)
               {
                    vector<vector<float> > detx;
		    vector<int> detx_id;
		    vector<list<Trajectory>::iterator> trax = q->second;
		    for(int i = 0; i < dets.size(); ++i)
		    {
			if (dets_used[i] == 0)
			{
			    detx.push_back(dets[i]);
			    detx_id.push_back(i);
			} 
		    }

                    int M = max(trax.size(), detx.size()); 
                    vector<vector<double> > cost(M, vector<double>(M, 10000));
                    vector<vector<double> > dist(M, vector<double>(M, 10000));
		    vector<vector<double> > sdist(M, vector<double>(M, 10000));
                    vector<vector<int>> A(M, vector<int>(M, 10000));

	            for(int i = 0; i < trax.size(); ++i)
                    {
			auto p = trax[i];    
                        vector<float> t = (p->Q).back(); 
	                float h00 = ht.at<double>(0,0);
                        float h01 = ht.at<double>(0,1);
                        float h02 = ht.at<double>(0,2);
                        float h10 = ht.at<double>(1,0);
                        float h11 = ht.at<double>(1,1);
                        float h12 = ht.at<double>(1,2);
                        float h20 = ht.at<double>(2,0);
                        float h21 = ht.at<double>(2,1);
                        float h22 = ht.at<double>(2,2);

                        float xw1 = (h00*(t[0]/6)+h01*(t[1]/6)+h02)/(h20*(t[0]/6)+h21*(t[1]/6)+h22)*6;
                        float yw1 = (h10*(t[0]/6)+h11*(t[1]/6)+h12)/(h20*(t[0]/6)+h21*(t[1]/6)+h22)*6;

                        float xw2 = (h00*(t[2]/6)+h01*(t[3]/6)+h02)/(h20*(t[2]/6)+h21*(t[3]/6)+h22)*6;
                        float yw2 = (h10*(t[2]/6)+h11*(t[3]/6)+h12)/(h20*(t[2]/6)+h21*(t[3]/6)+h22)*6;
	                
                        for (int j = 0; j < detx.size(); ++j)
                        {
                             dist[i][j] = box_dist(xw1, yw1, xw2, yw2, detx[j][0], detx[j][1], detx[j][2], detx[j][3]);
                             sdist[i][j] = size_dist(xw1, yw1, xw2, yw2, detx[j][0], detx[j][1], detx[j][2], detx[j][3]);
			     A[i][j] = appearance_dist(p->appearance, appearance[detx_id[j]]); 
                             cost[i][j] = 0.001*dist[i][j] +
                                          0.01*A[i][j];
                        }
                    }

                    vector<int> L;
                    vector<int> R;
                    mincostmatching(cost, L, R);

                    for (int i = 0; i < trax.size(); ++i)
                    {
                       if (L[i] >= detx.size()  || dist[i][L[i]] > 0.5 || sdist[i][L[i]] > 2 || sdist[i][L[i]] < 0.5) 
                          L[i] = -1;

                       if (L[i] >= 0)
                       {
                          trax[i]->appearance = appearance[detx_id[L[i]]];
                          trax[i]->extend(detx[L[i]][0],detx[L[i]][1], detx[L[i]][2], detx[L[i]][3], 1, 1);
                          dets_used[detx_id[L[i]]] = 1;
                       }
                       else 
                          trax[i]->mantain(ht, img);
                    }
	       }

               for (int i = 0; i < dets.size(); ++i)
               {
                   if (dets_used[i] == 0)
                   {
                       Trajectory x(dets[i][0], dets[i][1], dets[i][2], dets[i][3], 1, 1, tid);
                       x.appearance = appearance[i];
                       traks.push_back(x);
                       ++tid;
                   }
               }

          } 

          vector<vector<float> > rdets;
          vector<int> ids;
	  int probation = 1;

          for(list<Trajectory>::iterator p = traks.begin(); p != traks.end(); ++p )
          {
                    if (p->age >= probation)
                    {
                      vector<float> t = (p->Q).back();
                      rdets.push_back(t);
                      ids.push_back(p->id);
		    }
	  }

          for (int i = 0; i < rdets.size(); ++i)
          {
               cv::Rect rec;
               rec.x = rdets[i][0];
               rec.y = rdets[i][1];
               rec.width = rdets[i][2]-rdets[i][0];
               rec.height = rdets[i][3]-rdets[i][1];
               fprintf(fp_out, "%d,%d,%d,%d,%d,%d\n", int(frame_num), ids[i], int(2*rec.x), int(2*rec.y), 
			       int(2*rec.width), int(2*rec.height));	       

               char id_str[100];
               sprintf(id_str, "%d", ids[i]); 
               int cid = ids[i] % 1000;
               cv::putText(img, id_str, cv::Point(rec.x+rec.width/2, rec.y+rec.height/2), cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 2, 
			   CV_RGB(colors[cid][0], colors[cid][1], colors[cid][2]), 8);
               cv::rectangle(img, rec, CV_RGB(colors[cid][0], colors[cid][1], colors[cid][2]), 5);
	  }

          cv::imshow("Track", img);
	  ++frame_num;


          if ((char)27 == cv::waitKey(1) ) {  
	      cout << flush; 
	      break;
          }
     }

     fclose(fp_out);


     return 0;
}
