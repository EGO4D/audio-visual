#include <lcm/lcm-cpp.hpp>
#include "exlcm/image_stream_t.hpp"
#include "exlcm/boxes_image_t.hpp"
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
#include <mutex>
#include <thread>

#include <X11/Xlib.h>
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

static int demo_frame1 = 1;
static int demo_index1 = 0;
static float **predictions1;
static float *avg1;
static int demo_done = 0;
static int demo_total1 = 0;
double demo_time;

static int demo_frame2 = 1;
static int demo_index2 = 0;
static float **predictions2;
static float *avg2;
static int demo_total2 = 0;

static int demo_frame3 = 1;
static int demo_index3 = 0;
static float **predictions3;
static float *avg3;
static int demo_total3 = 0;


lcm::LCM lcmx;

const int H = 360;
const int W = 640;

std::mutex  myMutex_rgb1;
std::mutex  myMutex_rgb2;
std::mutex  myMutex_rgb3;
std::mutex  myMutex_rgb4;
std::mutex  myMutex_rgb5;
std::mutex  myMutex_rgb6;
std::mutex  myMutex_rgb7;
std::mutex  myMutex_rgb8;

cv::Mat image1 = cv::Mat::zeros(H, W, CV_8UC3);
cv::Mat image2 = cv::Mat::zeros(H, W, CV_8UC3);
cv::Mat image3 = cv::Mat::zeros(H, W, CV_8UC3);
cv::Mat image4 = cv::Mat::zeros(H, W, CV_8UC3);
cv::Mat image5 = cv::Mat::zeros(H, W, CV_8UC3);
cv::Mat image6 = cv::Mat::zeros(H, W, CV_8UC3);
cv::Mat image7 = cv::Mat::zeros(H, W, CV_8UC3);
cv::Mat image8 = cv::Mat::zeros(H, W, CV_8UC3);

int64_t current_time = 0;

bool has_new_frame = false;

class Handler1
{
    public:
        ~Handler1() {}

        void handleMessage(const lcm::ReceiveBuffer* rbuf,
                const std::string& chan,
                const exlcm::image_stream_t* msg)
        {
            //int i;
            //printf("Received message on channel \"%s\":\n", chan.c_str());
            //printf("  timestamp   = %lld\n", (long long)msg->timestamp);
            myMutex_rgb1.lock();
            cv::resize(cv::imdecode(cv::Mat(msg->data), 1), image1, cv::Size(W,H), 0, 0, cv::INTER_NEAREST);
            //cv::flip(image1, image1, 1);
            current_time = msg->timestamp;
            has_new_frame = true;
            myMutex_rgb1.unlock();
        }
};

class Handler2
{
    public:
        ~Handler2() {}

        void handleMessage(const lcm::ReceiveBuffer* rbuf,
                const std::string& chan,
                const exlcm::image_stream_t* msg)
        {
            //int i;
            //printf("Received message on channel \"%s\":\n", chan.c_str());
            //printf("  timestamp   = %lld\n", (long long)msg->timestamp);
            myMutex_rgb2.lock();
            cv::resize(cv::imdecode(cv::Mat(msg->data), 1), image2, cv::Size(W,H), 0, 0, cv::INTER_NEAREST);
            //cv::flip(image1, image1, 1);
            myMutex_rgb2.unlock();
        }
};

class Handler3
{
    public:
        ~Handler3() {}

        void handleMessage(const lcm::ReceiveBuffer* rbuf,
                const std::string& chan,
                const exlcm::image_stream_t* msg)
        {
            //int i;
            //printf("Received message on channel \"%s\":\n", chan.c_str());
            //printf("  timestamp   = %lld\n", (long long)msg->timestamp);
            myMutex_rgb3.lock();
            cv::resize(cv::imdecode(cv::Mat(msg->data), 1), image3, cv::Size(W,H), 0, 0, cv::INTER_NEAREST);
            //cv::flip(image1, image1, 1);
            myMutex_rgb3.unlock();
        }
};


void extract_image(int threadid) {

   Handler1 handlerObject1;
   lcmx.subscribe("rgb1", &Handler1::handleMessage, &handlerObject1);

   Handler2 handlerObject2;
   lcmx.subscribe("rgb2", &Handler2::handleMessage, &handlerObject2);

   Handler3 handlerObject3;
   lcmx.subscribe("rgb3", &Handler3::handleMessage, &handlerObject3);

   while(0 == lcmx.handle());

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

void remember_network1(network *neti)
{
    int i;
    int count = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            memcpy(predictions1[demo_index1] + count, net->layers[i].output, sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }
}

void remember_network2(network *neti)
{
    int i;
    int count = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            memcpy(predictions2[demo_index2] + count, net->layers[i].output, sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }
}

void remember_network3(network *neti)
{
    int i;
    int count = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            memcpy(predictions3[demo_index3] + count, net->layers[i].output, sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }
}


detection *avg_predictions1(network *net, int *nboxes)
{
    int i, j;
    int count = 0;
    fill_cpu(demo_total1, 0, avg1, 1);
    for(j = 0; j < demo_frame1; ++j){
        axpy_cpu(demo_total1, 1./demo_frame1, predictions1[j], 1, avg1, 1);
    }
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            memcpy(l.output, avg1 + count, sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }
    detection *dets = get_network_boxes(net, W, H, demo_thresh, demo_hier, 0, 1, nboxes);
    return dets;
}

detection *avg_predictions2(network *net, int *nboxes)
{
    int i, j;
    int count = 0;
    fill_cpu(demo_total2, 0, avg2, 1);
    for(j = 0; j < demo_frame2; ++j){
        axpy_cpu(demo_total2, 1./demo_frame2, predictions2[j], 1, avg2, 1);
    }
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            memcpy(l.output, avg2 + count, sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }
    detection *dets = get_network_boxes(net, W, H, demo_thresh, demo_hier, 0, 1, nboxes);
    return dets;
}

detection *avg_predictions3(network *net, int *nboxes)
{
    int i, j;
    int count = 0;
    fill_cpu(demo_total3, 0, avg3, 1);
    for(j = 0; j < demo_frame3; ++j){
        axpy_cpu(demo_total3, 1./demo_frame3, predictions3[j], 1, avg3, 1);
    }
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            memcpy(l.output, avg3 + count, sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }
    detection *dets = get_network_boxes(net, W, H, demo_thresh, demo_hier, 0, 1, nboxes);
    return dets;
}

void detect1(cv::Mat &imagex, char *out_channel)
{
     running = 1;
     float nms = .4;

     layer l = net->layers[net->n-1];
     image bimage = letterbox_image(mat_to_image(imagex.clone()), net->w, net->h);
     float *X = bimage.data;
     network_predict(net, X);
     remember_network1(net);
     detection *dets = 0;
     int nboxes = 0;
     dets = avg_predictions1(net, &nboxes);

     if (nms > 0) do_nms_obj(dets, nboxes, l.classes, nms);

     printf("\033[2J");
     printf("\033[1;1H");
     printf("\nFPS:%.1f\n",fps);
     printf("Objects:\n\n");
     cv::Mat im = imagex;

     exlcm::boxes_image_t boxes_image;

     std::vector<uchar> buf;
     bool code = cv::imencode(".jpg", im, buf);
     for (int i = 0; i < buf.size(); ++i)
        boxes_image.imdata.push_back(buf[i]);

     boxes_image.imlen = buf.size();

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

         if (cls == 1)
         {
            box b = dets[i].bbox;
            int left  = (b.x-b.w/2.)*im.cols;
            int right = (b.x+b.w/2.)*im.cols;
            int top   = (b.y-b.h/2.)*im.rows;
            int bot   = (b.y+b.h/2.)*im.rows;

            if(left < 0) left = 0;
            if(right > im.cols-1) right = im.cols-1;
            if(top < 0) top = 0;
            if(bot > im.rows-1) bot = im.rows-1;

            boxes_image.x1.push_back(left);
            boxes_image.y1.push_back(top);
            boxes_image.x2.push_back(right);
            boxes_image.y2.push_back(bot);
            boxes_image.id.push_back(cls);
            
            cv::rectangle(im, cv::Point(left, top), cv::Point(right, bot), Scalar(255,0,0), 5);
          }
     }          

     if (boxes_image.x1.size() == 0)
     {
         boxes_image.x1.push_back(-1);
         boxes_image.y1.push_back(-1);
         boxes_image.x2.push_back(-1);
         boxes_image.y2.push_back(-1);
         boxes_image.id.push_back(-1);
     }            

     boxes_image.boxlen = boxes_image.x1.size();

     lcmx.publish(out_channel, &boxes_image);
     
     //cv::imshow("Image", im);
     //cv::waitKey(1);
     //draw_detections(display, dets, nboxes, demo_thresh, demo_names, demo_alphabet, demo_classes);
     free_detections(dets, nboxes);
     demo_index1 = (demo_index1 + 1)%demo_frame1;
     running = 0;
}

void detect2(cv::Mat &imagex, char *out_channel)
{
     running = 1;
     float nms = .4;

     layer l = net->layers[net->n-1];
     image bimage = letterbox_image(mat_to_image(imagex.clone()), net->w, net->h);
     float *X = bimage.data;
     network_predict(net, X);
     remember_network2(net);
     detection *dets = 0;
     int nboxes = 0;
     dets = avg_predictions2(net, &nboxes);

     if (nms > 0) do_nms_obj(dets, nboxes, l.classes, nms);

     printf("\033[2J");
     printf("\033[1;1H");
     printf("\nFPS:%.1f\n",fps);
     printf("Objects:\n\n");
     cv::Mat im = imagex;

     exlcm::boxes_image_t boxes_image;

     std::vector<uchar> buf;
     bool code = cv::imencode(".jpg", im, buf);
     for (int i = 0; i < buf.size(); ++i)
        boxes_image.imdata.push_back(buf[i]);

     boxes_image.imlen = buf.size();

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

         if (cls == 1)
         {
            box b = dets[i].bbox;
            int left  = (b.x-b.w/2.)*im.cols;
            int right = (b.x+b.w/2.)*im.cols;
            int top   = (b.y-b.h/2.)*im.rows;
            int bot   = (b.y+b.h/2.)*im.rows;

            if(left < 0) left = 0;
            if(right > im.cols-1) right = im.cols-1;
            if(top < 0) top = 0;
            if(bot > im.rows-1) bot = im.rows-1;

            boxes_image.x1.push_back(left);
            boxes_image.y1.push_back(top);
            boxes_image.x2.push_back(right);
            boxes_image.y2.push_back(bot);
            boxes_image.id.push_back(cls);
            
            cv::rectangle(im, cv::Point(left, top), cv::Point(right, bot), Scalar(255,0,0), 5);
          }
     }          

     if (boxes_image.x1.size() == 0)
     {
         boxes_image.x1.push_back(-1);
         boxes_image.y1.push_back(-1);
         boxes_image.x2.push_back(-1);
         boxes_image.y2.push_back(-1);
         boxes_image.id.push_back(-1);
     }            

     boxes_image.boxlen = boxes_image.x1.size();

     lcmx.publish(out_channel, &boxes_image);
     
     //cv::imshow("Image", im);
     //cv::waitKey(1);
     //draw_detections(display, dets, nboxes, demo_thresh, demo_names, demo_alphabet, demo_classes);
     free_detections(dets, nboxes);
     demo_index2 = (demo_index2 + 1)%demo_frame2;
     running = 0;
}

void detect3(cv::Mat &imagex, char *out_channel)
{
     running = 1;
     float nms = .4;

     layer l = net->layers[net->n-1];
     image bimage = letterbox_image(mat_to_image(imagex.clone()), net->w, net->h);
     float *X = bimage.data;
     network_predict(net, X);
     remember_network3(net);
     detection *dets = 0;
     int nboxes = 0;
     dets = avg_predictions3(net, &nboxes);

     if (nms > 0) do_nms_obj(dets, nboxes, l.classes, nms);

     printf("\033[2J");
     printf("\033[1;1H");
     printf("\nFPS:%.1f\n",fps);
     printf("Objects:\n\n");
     cv::Mat im = imagex;

     exlcm::boxes_image_t boxes_image;

     std::vector<uchar> buf;
     bool code = cv::imencode(".jpg", im, buf);
     for (int i = 0; i < buf.size(); ++i)
        boxes_image.imdata.push_back(buf[i]);

     boxes_image.imlen = buf.size();

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

         if (cls == 1)
         {
            box b = dets[i].bbox;
            int left  = (b.x-b.w/2.)*im.cols;
            int right = (b.x+b.w/2.)*im.cols;
            int top   = (b.y-b.h/2.)*im.rows;
            int bot   = (b.y+b.h/2.)*im.rows;

            if(left < 0) left = 0;
            if(right > im.cols-1) right = im.cols-1;
            if(top < 0) top = 0;
            if(bot > im.rows-1) bot = im.rows-1;

            boxes_image.x1.push_back(left);
            boxes_image.y1.push_back(top);
            boxes_image.x2.push_back(right);
            boxes_image.y2.push_back(bot);
            boxes_image.id.push_back(cls);
            
            cv::rectangle(im, cv::Point(left, top), cv::Point(right, bot), Scalar(255,0,0), 5);
          }
     }          

     if (boxes_image.x1.size() == 0)
     {
         boxes_image.x1.push_back(-1);
         boxes_image.y1.push_back(-1);
         boxes_image.x2.push_back(-1);
         boxes_image.y2.push_back(-1);
         boxes_image.id.push_back(-1);
     }            

     boxes_image.boxlen = boxes_image.x1.size();

     lcmx.publish(out_channel, &boxes_image);
     
     //cv::imshow("Image", im);
     //cv::waitKey(1);
     //draw_detections(display, dets, nboxes, demo_thresh, demo_names, demo_alphabet, demo_classes);
     free_detections(dets, nboxes);
     demo_index3 = (demo_index3 + 1)%demo_frame3;
     running = 0;
}


void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
{
    //demo_frame = avg_frames;
    image **alphabet = load_alphabet();
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    demo_hier = hier;
    printf("Demo\n");
    net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);

    XInitThreads();
    std::thread image_thread(extract_image, 0);

    srand(2222222);

    int i;
    demo_total1 = size_network(net);
    predictions1 = (float **)calloc(demo_frame1, sizeof(float*));
    for (i = 0; i < demo_frame1; ++i){
       predictions1[i] = (float *)calloc(demo_total1, sizeof(float));
    }
    avg1 = (float *)calloc(demo_total1, sizeof(float));

    demo_total2 = size_network(net);
    predictions2 = (float **)calloc(demo_frame2, sizeof(float*));
    for (i = 0; i < demo_frame2; ++i){
       predictions2[i] = (float *)calloc(demo_total2, sizeof(float));
    }
    avg2 = (float *)calloc(demo_total2, sizeof(float));

    demo_total3 = size_network(net);
    predictions3 = (float **)calloc(demo_frame3, sizeof(float*));
    for (i = 0; i < demo_frame3; ++i){
       predictions3[i] = (float *)calloc(demo_total3, sizeof(float));
    }
    avg3 = (float *)calloc(demo_total3, sizeof(float));


    int count = 0;
    demo_time = what_time_is_it_now();

    while(!demo_done){
         if (has_new_frame)
         {
              has_new_frame = false;
              std::cout << count << std::endl;
              ++count; 
              detect1(image1, "yolo1");
              detect2(image2, "yolo2");
              detect3(image3, "yolo3");
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

