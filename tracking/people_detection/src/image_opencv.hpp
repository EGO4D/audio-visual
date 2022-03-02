#ifdef OPENCV

#include "stdio.h"
#include "stdlib.h"
#include "opencv2/opencv.hpp"
#include "image.h"

using namespace cv;

extern "C" {
IplImage *image_to_ipl(image im);
image ipl_to_image(IplImage* src);
Mat image_to_mat(image im);
image mat_to_image(Mat m);
void *open_video_stream(const char *f, int c, int w, int h, int fps);
image get_image_from_stream(void *p);
image load_image_cv(char *filename, int channels);
int show_image_cv(image im, const char* name, int ms);
}

#endif
