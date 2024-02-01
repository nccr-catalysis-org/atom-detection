/*
   @author : Romain Graux
   @date : 2023 April 03, 15:30:30
   @last modified : 2023 April 03, 18:34:08
   */

#include <iostream>
#include <algorithm>

using namespace std;

extern "C" {
    void median_filter(float* data, int width, int height, int window_size, float* out);
    void reflect_borders(float *data, int width, int height, int span, float *out);
}


void reflect_borders(float *data, int width, int height, int span, float *out){
    int out_width = width + 2*span;
    int out_height = height + 2*span;
    // First copy the same data but with a border of span pixels
    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
            out[(i+span)*(out_width) + j + span] = data[i*width + j];
        }
    }

    // Then reflect the top and bottom borders
    for (int j=0; j<width; j++){
        for (int h=0; h<span; h++){
            out[(span-h-1)*out_width + j + span] = out[(span+h)*out_width + j + span];
            out[(out_height-span+h)*out_width + j + span] = out[(out_height-span-h-1)*out_width + j + span];
        }
    }

    // Then reflect the left and right borders
    for(int i=0; i<out_height; i++){
        for(int w=0; w<span; w++){
            out[i*out_width + span - w - 1] = out[i*out_width + span + w];
            out[i*out_width + width + span + w] = out[i*out_width + width + span - w - 1];
        }
    }
}


void median_filter(float* data, int width, int height, int windowSize, float* out){
    int span = windowSize/2;
    int padded_width = width + 2*span;

    float* window = new float[windowSize*windowSize];
    for (int y = span; y < height + span; y++){
        for (int x = span; x < width + span; x++){
            for (int i = 0; i < windowSize; i++){
                for (int j = 0; j < windowSize; j++){
                    window[i*windowSize + j] = data[(y-span+i)*padded_width + (x-span+j)];
                }
            }
            std::nth_element(window, window + windowSize*windowSize/2, window + windowSize*windowSize);
            out[(y-span)*width + (x-span)] = window[windowSize*windowSize/2];
        }
    }
    delete[] window;
}


int main(){
    int width = 4;
    int height = 4;
    int window_size = 2;
    float* data = new float[width*height];
    float* out = new float[(width+2*window_size/2)*(height+2*window_size/2)];
    for(int i=0; i<width*height; i++){
        data[i] = i;
    }
    reflect_borders(data, width, height, window_size/2, out);
    for (int i=0; i<height+2*window_size/2; i++){
        for (int j=0; j<width+2*window_size/2; j++){
            cout << out[i*(width+2*window_size/2) + j] << " ";
        }
        cout << endl;
    }
    return 0;
}
