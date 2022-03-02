/** 
 * File:    mrc.cpp 
 * 
 * Summary of File: active speaker detection using mouth region classification (mrc).          
 * Usage: ./mrc directory_of_ego4d_videos directory_of_global_tracking_results index_of_the_video active_speaker_model
 * 
 */

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <stdlib.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <deque>
#include <list>

#include <stdio.h>
#include <ctime>

#include <torch/script.h> 

using namespace std;

torch::jit::script::Module module_head_pose;
torch::jit::script::Module module_speaker;
torch::jit::script::Module module_gaze;

void pred_head_pose(const cv::Mat &im, vector<float> &v1, vector<float> &v2)
{

           cv::Mat im2;
	   cv::resize(im, im2, cv::Size(128, 128));
           cv::Mat image;	   
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
           auto res = module_head_pose.forward(inputs).toTuple();
           at::Tensor result1 = res->elements()[0].toTensor();
           at::Tensor result2 = res->elements()[1].toTensor();

           auto r1 = result1.to(at::kCPU);
           auto r2 = result2.to(at::kCPU);

           auto out1 = r1.accessor<float, 2>();
           for (int i = 0; i < 136; ++i)
           {
		    v1.push_back(out1[0][i]);
	   }

           auto out2 = r2.accessor<float, 2>();
           for (int i = 0; i < 3; ++i)
           {
		   v2.push_back(2*out2[0][i]-1);
           }

}

void pred_speaker(const cv::Mat &im, vector<float> &v2)
{

           cv::Mat im2;
           cv::resize(im, im2, cv::Size(128, 128));
           cv::Mat image;	   
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
           auto res = module_speaker.forward(inputs);
           at::Tensor result2 = res.toTensor();

           auto r2 = result2.to(at::kCPU);

           auto out2 = r2.accessor<float, 2>();
           for (int i = 0; i < 3; ++i)
           {
                   v2.push_back(out2[0][i]);
           }

}

void pred_gaze(const cv::Mat &im, vector<float> &v2)
{

           cv::Mat im2;
           cv::resize(im, im2, cv::Size(128, 128));
           cv::Mat image;	   
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
           auto res = module_gaze.forward(inputs);
           at::Tensor result2 = res.toTensor();

           auto r2 = result2.to(at::kCPU);

           auto out2 = r2.accessor<float, 2>();
           for (int i = 0; i < 3; ++i)
           {
                   v2.push_back(out2[0][i]);
           }

}



int main(int argn, char **argv )
{
     module_head_pose = torch::jit::load("../../../../models/face_points_head_pose.pt");
     module_speaker = torch::jit::load(argv[4]);
     module_gaze = torch::jit::load("../../../../models/gaze.pt");
     module_head_pose.to(at::kCUDA);
     module_speaker.to(at::kCUDA);
     module_gaze.to(at::kCUDA);     
   
     string video_dir(argv[1]);
     char *tracking_results_dir = argv[2];
     int video_num = stoi(argv[3]);
     vector<string> video_file_names;
     string line;    
     ifstream fin("v.txt");
     while(getline(fin,line))
     {
         video_file_names.push_back(video_dir + "/" + line);	 
     }     
     fin.close();

     map<int, vector<vector<int>> > boxes;
     char box_fname[1000];
     sprintf(box_fname, "%s/%d.txt", tracking_results_dir, video_num);
     
     FILE *fpbox = fopen(box_fname, "rt");
     int a, b, c, d, e, f;
     while (fscanf(fpbox, "%d %d %d %d %d %d", &a, &b, &c, &d, &e, &f) != EOF)
     {
        if (boxes.find(a) == boxes.end())
            boxes[a] = vector< vector<int> >();

        boxes[a].push_back({b, c, d, e, f});
     }	

     fclose(fpbox);
     
     cv::Mat frame;
     cv::VideoCapture capture(video_file_names[video_num]);
     cout << video_file_names[video_num];
     if (!capture.isOpened())
     {
         cerr << "ERROR: Can't open " << video_file_names[video_num] << endl;
         return 1;
     }

     FILE *fp_out = fopen("result.txt", "wt");
     for (;;)
     {
        static int frame_num = 0;
	
        capture >> frame;

        if (frame.empty()) break;
	auto img = frame.clone();
	auto img_draw = frame.clone();

	if (boxes.find(frame_num) != boxes.end() )
        {

          for (int bnum = 0; bnum < boxes[frame_num].size(); ++bnum)
	  {
             int pid = boxes[frame_num][bnum][0];
             int x1 = boxes[frame_num][bnum][1];
             int y1 = boxes[frame_num][bnum][2];
             int x2 = boxes[frame_num][bnum][3];
             int y2 = boxes[frame_num][bnum][4];
	     
             if (x1 < 0)
                x1 = 0;

             if (x2 >= img.cols-1)
                x2 = img.cols-1;

             if (y1 < 0)
                y1 = 0;

             if (y2 >= img.rows-1)
                y2 = img.rows-1;
	  
	     if (x2 <= x1 || y2 <= y1)
		  continue;   
	      
	     vector<float> v1;
             vector<float> v2;
	     cv::Mat imcrop = img(cv::Rect(x1, y1, x2-x1+1, y2-y1+1)).clone();
             pred_head_pose(imcrop, v1, v2);

             vector<float> v3;
             pred_gaze(imcrop, v3);
	    
	     if (v3[2] > 0.3) 
             {
                 fprintf(fp_out, "%d %d %d %d %d %d %d %f\n", int(frame_num), pid, x1, y1, x2, y2, 0, 0.0);		     
		 continue;
	     } 

	     vector<float> kx;
	     vector<float> ky;
	     for (int k = 0; k < 68; ++k)
	     {
		     kx.push_back((x2-x1)*v1[2*k]+x1);
		     ky.push_back((y2-y1)*v1[2*k+1]+y1);
	     }


             for (int k = 0; k < 68; ++k)
                 cv::circle(img_draw, cv::Point(int(kx[k]), int(ky[k])), 2, cv::Scalar(0,0,255), -1);

             int x0 = int((x1+x2)/2.0);
             int y0 = int((y1+y2)/2.0);

             cv::line(img_draw, cv::Point(x0,y0), cv::Point(int(x0+100*v2[0]), int(y0+100*v2[1])), cv::Scalar(255,0,0), 3);
	     
	     float mx_min = 1e6;
	     float mx_max = -1e6;
	     float my_min = 1e6;
	     float my_max = -1e6;

	     for(int k = 48; k < 68; ++k)
	     {
		 if (kx[k] < mx_min) mx_min = kx[k];
                 if (kx[k] > mx_max) mx_max = kx[k];
                 if (ky[k] < my_min) my_min = ky[k];
                 if (ky[k] > my_max) my_max = ky[k];
	     }

	     float sz = max(mx_max-mx_min, my_max-my_min);
             float mx0 = (mx_min + mx_max) / 2.0;
             float my0 = (my_min + my_max) / 2.0;

             int mx1 = int(mx0 - sz * 0.6);
             int my1 = int(my0 - sz * 0.6);
             int mx2 = int(mx0 + sz * 0.6);
             int my2 = int(my0 + sz * 0.6);

             if (mx1 < 0)
                mx1 = 0;

             if (mx2 >= img.cols-1)
                mx2 = img.cols-1;

             if (my1 < 0)
                my1 = 0;

             if (my2 >= img.rows-1)
                my2 = img.rows-1;


	     int speak_a = 0;
	     float speak_c = 0;
             if (mx2 > mx1 && my2 > my1)
	     {
                  vector<float> v3;
                  cv::Mat immouth = img(cv::Rect(mx1, my1, mx2-mx1+1, my2-my1+1)).clone();
                  pred_speaker(immouth, v3);
		  speak_c = 1-v3[0];
                  if (1-v3[0] > 0.97)	
		  {	  
                     cv::rectangle(img_draw, cv::Point(x1, y1), cv::Point(x2, y2), 
				   cv::Scalar(255,0,255), 20);
		     speak_a = 1;
		  }
                  else
                     cv::rectangle(img_draw, cv::Point(x1, y1), cv::Point(x2, y2),
                                   cv::Scalar(255,0,0), 2);			  
			  
	     }

	     fprintf(fp_out, "%d %d %d %d %d %d %d %f\n", int(frame_num), pid, x1, y1, x2, y2, speak_a, speak_c);
   
          }
	}


        cv::Mat img2;
        cv::resize(img_draw, img2, cv::Size(img_draw.cols/2, img_draw.rows/2));
	
	cv::imshow("Active speaker", img2);
        cv::waitKey(1);
	++frame_num; 
     }

    fclose(fp_out);

    return 0;
}
