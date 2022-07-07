#include <stdio.h>

#include <inc_irit/irit_sm.h>
#include <inc_irit/misc_lib.h>

void main()
{
    int i;
    double R;
    IrtImgRGBAPxlStruct *Pixels;
    FILE
        *f = fopen("xPhys.txt", "r");

    Pixels = (IrtImgRGBAPxlStruct *)
                           IritMalloc(sizeof(IrtImgRGBAPxlStruct *) * 500000);

    for (i = 0; i < 500000 && !feof(f); i++) {
        fscanf(f, "%lf", &R);
	Pixels[i].r = R > 0 ? 255 : 0;
	Pixels[i].g = R > 0 ? 255 : 0;
	Pixels[i].b = R > 0 ? 255 : 0;
	Pixels[i].a = 0;

	if (!IRIT_APX_EQ(R, 0.0) && !IRIT_APX_EQ(R, 1.0)) {
	    fprintf(stderr, "Found value %7f\n", R);
	}
    }

    IrtImgWriteImg("xPhys.ppm", Pixels, 1000, 500, FALSE);
}
