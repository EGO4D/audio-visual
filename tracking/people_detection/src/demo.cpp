#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include "image_opencv.hpp"
#include <sys/time.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <time.h>

#define DEMO 1

#ifdef OPENCV

char channel_name[1000];
char out_channel_name[1000];
static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static network *net;
static float fps = 0;
static float demo_thresh = 0;
static float demo_hier = .5;
static int running = 0;

static int demo_frame = 1;
static int demo_index = 0;
static float **predictions;
static float *avg;
static int demo_done = 0;
static int demo_total = 0;
double demo_time;

int H = 360;
int W = 640;

cv::Mat image1 = cv::Mat::zeros(H, W, CV_8UC3);

int64_t current_time = 0;

bool has_new_frame = false;

void load_mat(char filename[], Mat & A)
{
     FILE *fp = fopen(filename, "r");
     if (!fp)
     {
         std::cout << "cannot open " << filename << std::endl;
         exit(0);
     }
     for (int i = 0; i < A.rows; i++)
     for (int j = 0; j < A.cols; j++)
     {
         float t;
         fscanf(fp, "%f", &t);
         A.at<float>(i,j) = t;
     }
     fclose(fp);
}


detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num);

int size_network(network *net)
{
    int i;
    int count = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            count += l.outputs;
        }
    }
    return count;
}

void remember_network(network *neti)
{
    int i;
    int count = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            memcpy(predictions[demo_index] + count, net->layers[i].output, sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }
}

detection *avg_predictions(network *net, int *nboxes)
{
    int i, j;
    int count = 0;
    fill_cpu(demo_total, 0, avg, 1);
    for(j = 0; j < demo_frame; ++j){
        axpy_cpu(demo_total, 1./demo_frame, predictions[j], 1, avg, 1);
    }
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            memcpy(l.output, avg + count, sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }
    detection *dets = get_network_boxes(net, W, H, demo_thresh, demo_hier, 0, 1, nboxes);
    return dets;
}

void detect(cv::Mat &imagex, char *out_channel)
{
     H = imagex.rows;
     W = imagex.cols;     
     running = 1;
     float nms = 0.4;

     layer l = net->layers[net->n-1];
     image cimage = mat_to_image(imagex.clone());
     image bimage = letterbox_image(cimage, net->w, net->h);
     float *X = bimage.data;
     network_predict(net, X);
     remember_network(net);
     detection *dets = 0;
     int nboxes = 0;
     dets = avg_predictions(net, &nboxes);

     do_nms_obj(dets, nboxes, l.classes, nms);

     free_image(cimage);
     free_image(bimage);

     printf("\033[2J");
     printf("\033[1;1H");
     printf("\nFPS:%.1f\n",fps);
     printf("Objects:\n\n");
     cv::Mat im = imagex;

     for (int i = 0; i < nboxes; ++i)
     {
         char labelstr[4096] = {0};
         int cls = -1;
         for(int j = 0; j < demo_classes; ++j){
             if (dets[i].prob[j] > demo_thresh){
                 if (cls < 0) {
                     strcat(labelstr, demo_names[j]);
                     cls = j;
                 } else {
                     strcat(labelstr, ", ");
                     strcat(labelstr, demo_names[j]);
                 }
                 printf("%s: %.0f%%\n", demo_names[j], dets[i].prob[j]*100);
             }
         }

	 if (cls == 0 || cls == 1) {
                 box b = dets[i].bbox;
                 int left  = (b.x-b.w/2.)*im.cols;
                 int right = (b.x+b.w/2.)*im.cols;
                 int top   = (b.y-b.h/2.)*im.rows;
                 int bot   = (b.y+b.h/2.)*im.rows;
        
                 if(left < 0) left = 0;
                 if(right > im.cols-1) right = im.cols-1;
                 if(top < 0) top = 0;
                 if(bot > im.rows-1) bot = im.rows-1;
        
	 }
     }          

     cv::imshow("Image", im);
     cv::waitKey(1);
     free_detections(dets, nboxes);
     demo_index = 0;
     running = 0;
}


void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
{
    image **alphabet = load_alphabet();
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    demo_hier = hier;
    printf("Demo\n");
    net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);

    srand(2222222);

    int i;
    demo_total = size_network(net);
    predictions = (float **)calloc(demo_frame, sizeof(float*));
    for (i = 0; i < demo_frame; ++i){
       predictions[i] = (float *)calloc(demo_total, sizeof(float));
    }
    avg = (float *)calloc(demo_total, sizeof(float));

    int count = 0;
    demo_time = what_time_is_it_now();

    while(!demo_done){
         if (has_new_frame)
         {
              has_new_frame = false;
              std::cout << count << std::endl;
              ++count; 
              detect(image1, "yolo");
         }
         cv::waitKey(1);
    }
}

#else
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg, float hier, int w, int h, int frames, int fullscreen)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif

