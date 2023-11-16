/* CS580 Homework 3 */

#include	"stdafx.h"
#include	"stdio.h"
#include	"math.h"
#include	"Gz.h"
#include	"rend.h"
#include <algorithm>
#include <iostream>

#define PI (float) 3.14159265358979323846
#define EDGE_THRESHOLD (float) 1250

int GzRender::GzRotXMat(float degree, GzMatrix mat)
{
/* HW 3.1
// Create rotate matrix : rotate along x axis
// Pass back the matrix using mat value
*/

	float radian = degree * PI / 180.0;

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			mat[i][j] = 0;
		}
	}

	mat[0][0] = 1;
	mat[1][1] = cos(radian);
	mat[1][2] = -sin(radian);
	mat[2][1] = sin(radian);
	mat[2][2] = cos(radian);
	mat[3][3] = 1;
	return GZ_SUCCESS;
}

int GzRender::GzRotYMat(float degree, GzMatrix mat)
{
	/* HW 3.2
	// Create rotate matrix : rotate along y axis
	// Pass back the matrix using mat value
	*/
	float radian = degree * PI / 180.0;

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			mat[i][j] = 0;
		}
	}

	mat[0][0] = cos(radian);
	mat[0][2] = sin(radian);
	mat[1][1] = 1;
	mat[2][0] = -sin(radian);
	mat[2][2] = cos(radian);
	mat[3][3] = 1;

	return GZ_SUCCESS;
}

int GzRender::GzRotZMat(float degree, GzMatrix mat)
{
	/* HW 3.3
	// Create rotate matrix : rotate along z axis
	// Pass back the matrix using mat value
	*/
	float radian = degree * PI / 180.0;

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			mat[i][j] = 0;
		}
	}

	mat[0][0] = cos(radian);
	mat[0][1] = -sin(radian);
	mat[1][0] = sin(radian);
	mat[1][1] = cos(radian);
	mat[2][2] = 1;
	mat[3][3] = 1;

	return GZ_SUCCESS;
}

int GzRender::GzTrxMat(GzCoord translate, GzMatrix mat)
{
	/* HW 3.4
	// Create translation matrix
	// Pass back the matrix using mat value
	*/
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			if (i == j)
			{
				mat[i][j] = 1;
			}
			else {
				mat[i][j] = 0;
			}
		}
	}
	mat[0][3] = translate[0];
	mat[1][3] = translate[1];
	mat[2][3] = translate[2];

	return GZ_SUCCESS;
}


int GzRender::GzScaleMat(GzCoord scale, GzMatrix mat)
{
	/* HW 3.5
	// Create scaling matrix
	// Pass back the matrix using mat value
	*/
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			if (i == j && i < 3)
			{
				mat[i][j] = scale[i];
			}
			else {
				mat[i][j] = 0;
			}
		}
	}

	mat[3][3] = 1;
	return GZ_SUCCESS;
}


GzRender::GzRender(int xRes, int yRes)
{
	/* HW1.1 create a framebuffer for MS Windows display:
	 -- set display resolution
	 -- allocate memory for framebuffer : 3 bytes(b, g, r) x width x height
	 -- allocate memory for pixel buffer
	 */
	xres = xRes;
	yres = yRes;
	int resolution = xres * yres;
	framebuffer = new char[3 * sizeof(char) * resolution];
	pixelbuffer = new GzPixel[resolution];

	/* HW 3.6
	- setup Xsp and anything only done once
	- init default camera
	*/

	// set initial lookat = [0,0,0]
	// initial worldup = [0,1,0]
	for (int i = 0; i < 3; i++)
	{
		m_camera.lookat[i] = 0;
		m_camera.worldup[i] = 0;
	}

	m_camera.worldup[1] = 1;

	//set default position
	m_camera.position[0] = DEFAULT_IM_X;
	m_camera.position[1] = DEFAULT_IM_Y;
	m_camera.position[2] = DEFAULT_IM_Z;

	//set default FOV
	m_camera.FOV = DEFAULT_FOV;
	//set default matlevel
	numlights = 0;
	matlevel = -1;

	//init Xsp
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			Xsp[i][j] = 0;
		}
	}
	Xsp[0][0] = xres / 2;
	Xsp[0][3] = xres / 2;
	Xsp[1][1] = -yres / 2;
	Xsp[1][3] = yres / 2;
	Xsp[2][2] = MAXINT;
	Xsp[3][3] = 1;
}

GzRender::~GzRender()
{
	/* HW1.2 clean up, free buffer memory */
	delete[] framebuffer;
	delete[] pixelbuffer;
}

int GzRender::GzDefault()
{
	/* HW1.3 set pixel buffer to some default values - start a new frame */
	int resolution = xres * yres;
	for (int i = 0; i <= resolution; i++)
	{
		pixelbuffer[i] = { 4095,4095,4095,1,MAXINT };
		framebuffer[3 * i] = (char)4095;
		framebuffer[3 * i + 1] = (char)4095;
		framebuffer[3 * i + 2] = (char)4095;
	}
	return GZ_SUCCESS;
}


//Helper to calculate the norm of a 3D-vector
float calculateNorm(GzCoord coord) {
	float res = 0.0;
	for (int i = 0; i < 3; i++)
	{
		res += coord[i] * coord[i];
	}
	res = sqrt(res);
	return res;
}

//Helper to calculate dot product of 2 3D-vector
float dotProduct(GzCoord a, GzCoord b) {
	float res = 0.0;
	for (int i = 0; i < 3; i++)
	{
		res += a[i] * b[i];
	}
	return res;
}

//Helper 
bool approximatelyEqual(float a, float b, float epsilon = 1e-4) {
	// Check if the absolute difference between a and b is less than epsilon
	return fabs(a - b) < epsilon;
}

//Helper
float* crossProduct(float vec1[], float vec2[]) {
	float res[3];

	res[0] = vec1[1] * vec2[2] - vec1[2] * vec2[1];
	res[1] = vec1[2] * vec2[0] - vec1[0] * vec2[2];
	res[2] = vec1[0] * vec2[1] - vec1[1] * vec2[0];

	return res;
}

//Helper 
float* VecCrossProduct(GzCoord vec1, GzCoord vec2) {
	float res[3];

	res[0] = vec1[1] * vec2[2] - vec1[2] * vec2[1];
	res[1] = vec1[2] * vec2[0] - vec1[0] * vec2[2];
	res[2] = vec1[0] * vec2[1] - vec1[1] * vec2[0];

	return res;
}
//Helper
/*
int getcolor(GzRender* render, GzCoord normal, GzColor color) {
	GzCoord E = { 0,0,-1 };
	GzCoord R;
	float NdotE, NdotL, RdotE;
	GzColor diffuseIntensity = {0,0,0}; GzColor specIntensity = {0,0,0};
	NdotE = dotProduct(normal,E);

	GzCoord newN = { 0, 0, 0 };

	for (int i = 0; i < render->numlights; i++) {
		NdotL = dotProduct(normal, render->lights[i].direction);
		if (NdotL > 0 && NdotE > 0) {
			R[0] = (2 * (NdotL) * normal[0]) - (render->lights[i].direction[0]);
			R[1] = (2 * (NdotL) * normal[1]) - (render->lights[i].direction[1]);
			R[2] = (2 * (NdotL) * normal[2]) - (render->lights[i].direction[2]);
		}
		else if (NdotL < 0 && NdotE < 0) { // both negative flip N, calculate new NdotL
			newN[0] = (-1 * normal[0]);
			newN[1] = (-1 * normal[1]);
			newN[2] = (-1 * normal[2]);
			NdotL = dotProduct(newN, render->lights[i].direction);
			R[0] = (2 * (NdotL)*newN[0]) - (render->lights[i].direction[0]);
			R[1] = (2 * (NdotL)*newN[1]) - (render->lights[i].direction[1]);
			R[2] = (2 * (NdotL)*newN[2]) - (render->lights[i].direction[2]);
		}
		else
		{
			continue;
		}

		float modR = sqrt((R[0] * R[0]) + (R[1] * R[1]) + (R[2] * R[2]));

		R[0] = R[0] / modR;
		R[1] = R[1] / modR;
		R[2] = R[2] / modR;

		RdotE = min(1, max(0, dotProduct(R, E)));

		diffuseIntensity[RED] += ((NdotL) * (render->lights[i].color[RED]));
		diffuseIntensity[GREEN] += ((NdotL) * (render->lights[i].color[GREEN]));
		diffuseIntensity[BLUE] += ((NdotL) * (render->lights[i].color[BLUE]));

		specIntensity[RED] += ((pow(RdotE, render->spec)) * (render->lights[i].color[RED]));
		specIntensity[GREEN] += ((pow(RdotE, render->spec)) * (render->lights[i].color[GREEN]));
		specIntensity[BLUE] += ((pow(RdotE, render->spec)) * (render->lights[i].color[BLUE]));

	}

	color[RED] = (render->Ks[RED] * specIntensity[RED]) + (render->Kd[RED] * diffuseIntensity[RED]) + (render->Ka[RED] * render->ambientlight.color[RED]);
	color[GREEN] = (render->Ks[GREEN] * specIntensity[GREEN]) + (render->Kd[GREEN] * diffuseIntensity[GREEN]) + (render->Ka[GREEN] * render->ambientlight.color[GREEN]);
	color[BLUE] = (render->Ks[BLUE] * specIntensity[BLUE]) + (render->Kd[BLUE] * diffuseIntensity[BLUE]) + (render->Ka[BLUE] * render->ambientlight.color[BLUE]);

	return GZ_SUCCESS;
}
*/

//Helper
int getcolor(GzRender* render, GzCoord normal, GzColor color) {
	GzCoord E = { 0, 0, -1 };
	GzCoord R;
	float NdotE, NdotL, RdotE;
	GzColor diffuseIntensity = { 0, 0, 0 };
	NdotE = dotProduct(normal, E);
	GzCoord newN = { 0, 0, 0 };

	for (int i = 0; i < render->numlights; i++) {
		NdotL = dotProduct(normal, render->lights[i].direction);
		if (NdotL > 0 && NdotE > 0) {
			R[0] = (2 * (NdotL)*normal[0]) - (render->lights[i].direction[0]);
			R[1] = (2 * (NdotL)*normal[1]) - (render->lights[i].direction[1]);
			R[2] = (2 * (NdotL)*normal[2]) - (render->lights[i].direction[2]);
		}
		else if (NdotL < 0 && NdotE < 0) { // both negative flip N, calculate new NdotL
			newN[0] = (-1 * normal[0]);
			newN[1] = (-1 * normal[1]);
			newN[2] = (-1 * normal[2]);
			NdotL = dotProduct(newN, render->lights[i].direction);
			R[0] = (2 * (NdotL)*newN[0]) - (render->lights[i].direction[0]);
			R[1] = (2 * (NdotL)*newN[1]) - (render->lights[i].direction[1]);
			R[2] = (2 * (NdotL)*newN[2]) - (render->lights[i].direction[2]);
		}
		else
		{
			continue;
		}

		float modR = sqrt((R[0] * R[0]) + (R[1] * R[1]) + (R[2] * R[2]));

		R[0] = R[0] / modR;
		R[1] = R[1] / modR;
		R[2] = R[2] / modR;

		RdotE = min(1, max(0, dotProduct(R, E)));

		diffuseIntensity[RED] += ((NdotL) * (render->lights[i].color[RED]));
		diffuseIntensity[GREEN] += ((NdotL) * (render->lights[i].color[GREEN]));
		diffuseIntensity[BLUE] += ((NdotL) * (render->lights[i].color[BLUE]));
	}

	// 定义一个更好的颜色调色板
	GzColor brightColor = { 1.2, 1.15, 1.1 }; // 带有一丝温暖的亮色调
	GzColor darkColor = { 0.4, 0.3, 0.35 };  // 暗色调，用于阴影

	// 应用卡通着色的分段光照效果
		float threshold = 0.5; // 设定一个阈值
	for (int i = 0; i < 3; i++) {
		if (diffuseIntensity[i] > threshold) {
			color[i] = brightColor[i]; // 明亮部分
		}
		else {
			color[i] = darkColor[i]; // 暗淡部分
		}
	}

	color[RED] = (render->Kd[RED] * color[RED]) + (render->Ka[RED] * render->ambientlight.color[RED]);
	color[GREEN] = (render->Kd[GREEN] * color[GREEN]) + ( render->Ka[GREEN] * render->ambientlight.color[GREEN]);
	color[BLUE] = (render->Kd[BLUE] * color[BLUE]) + (render->Ka[BLUE] * render->ambientlight.color[BLUE]);


	return GZ_SUCCESS;
}

//Helper Simple edge detection
void EdgeDetection(GzRender* render) {
	int width = render->xres;
	int height = render->yres;
	GzPixel* pBuffer = render->pixelbuffer;
	GzPixel* newPbuffer = new GzPixel[width * height];

	// 遍历每个像素
	for (int j = 1; j < height - 1; j++) {
		for (int i = 1; i < width - 1; i++) {
			// 获取周围像素的颜色值
			int sumRed = 0;
			int sumGreen = 0;
			int sumBlue = 0;
			for (int y = -1; y <= 1; y++) {
				for (int x = -1; x <= 1; x++) {
					GzPixel currentPixel = pBuffer[(j + y) * width + (i + x)];
					sumRed += currentPixel.red;
					sumGreen += currentPixel.green;
					sumBlue += currentPixel.blue;
				}
			}

			// 计算平均颜色值
			int avgRed = sumRed / 9;
			int avgGreen = sumGreen / 9;
			int avgBlue = sumBlue / 9;

			// 比较当前像素与平均颜色值的差异
			GzPixel centerPixel = pBuffer[j * width + i];
			if (abs(centerPixel.red - avgRed) > EDGE_THRESHOLD ||
				abs(centerPixel.green - avgGreen) > EDGE_THRESHOLD ||
				abs(centerPixel.blue - avgBlue) > EDGE_THRESHOLD) {
				// 如果差异超过阈值，则认为是边缘，并将其设为黑色
				newPbuffer[j * width + i] = { 0, 0, 0, 255, centerPixel.z };
			}
			else {
				// 否则，保留原颜色
				newPbuffer[j * width + i] = centerPixel;
			}
		}
	}

	// 将新的帧缓冲区内容复制回原帧缓冲区
	memcpy(pBuffer, newPbuffer, sizeof(GzPixel) * width * height);
	delete[] newPbuffer;
}

// Helper Color simplification
// Helper function to quantize a single color channel
GzIntensity QuantizeChannel(GzIntensity channel, int levels) {
	float scaleFactor = 4095.0f / (levels - 1); // 4095 is the max intensity for a channel
	float quantizedValue = round(channel / scaleFactor) * scaleFactor;
	return static_cast<GzIntensity>(quantizedValue);
}
// Helper Color simplification
void ApplyColorQuantization(int colorLevels, GzRender* render) {
	int resolution = render->xres * render->yres;
	for (int i = 0; i < resolution; i++) {
		GzPixel& pixel = render->pixelbuffer[i];
		pixel.red = QuantizeChannel(pixel.red, colorLevels);
		pixel.green = QuantizeChannel(pixel.green, colorLevels);
		pixel.blue = QuantizeChannel(pixel.blue, colorLevels);
	}
}



int GzRender::GzBeginRender()
{
	/* HW 3.7
	- setup for start of each frame - init frame buffer color,alpha,z
	- compute Xiw and projection xform Xpi from camera definition
	- init Ximage - put Xsp at base of stack, push on Xpi and Xiw
	- now stack contains Xsw and app can push model Xforms when needed
	*/
	GzDefault();

	// compute Xiw
	GzCoord cl, up_prime, camera_X, camera_Y, camera_Z;
	for (int i = 0; i < 3; i++)
	{
		cl[i] = m_camera.lookat[i] - m_camera.position[i];
	}

	float norm_cl = calculateNorm(cl);
	// Z coord
	for (int i = 0; i < 3; i++)
	{
		camera_Z[i] = cl[i] / norm_cl;
	}

	float updotZ = dotProduct(m_camera.worldup, camera_Z);
	for (int i = 0; i < 3; i++) {
		up_prime[i] = m_camera.worldup[i] - (updotZ * camera_Z[i]);
	}

	//Y coord
	float norm_up_prime = calculateNorm(up_prime);
	for (int i = 0; i < 3; i++)
	{
		camera_Y[i] = up_prime[i] / norm_up_prime;
	}

	float* crossP = VecCrossProduct(camera_Y, camera_Z); //(A,B,C)
	//X coord
	for (int i = 0; i < 3; i++)
	{
		camera_X[i] = crossP[i];
	}

	//init Xiw
	m_camera.Xiw[0][0] = camera_X[0]; m_camera.Xiw[0][1] = camera_X[1]; m_camera.Xiw[0][2] = camera_X[2];
	m_camera.Xiw[0][3] = -1 * dotProduct(camera_X, m_camera.position);
	m_camera.Xiw[1][0] = camera_Y[0]; m_camera.Xiw[1][1] = camera_Y[1]; m_camera.Xiw[1][2] = camera_Y[2];
	m_camera.Xiw[1][3] = -1 * dotProduct(camera_Y, m_camera.position);
	m_camera.Xiw[2][0] = camera_Z[0]; m_camera.Xiw[2][1] = camera_Z[1]; m_camera.Xiw[2][2] = camera_Z[2];
	m_camera.Xiw[2][3] = -1 * dotProduct(camera_Z, m_camera.position);
	m_camera.Xiw[3][0] = 0; m_camera.Xiw[3][1] = 0; m_camera.Xiw[3][2] = 0; m_camera.Xiw[3][3] = 1;

	//compute Xpi
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			if (i == j) {
				m_camera.Xpi[i][j] = 1;
			}
			else {
				m_camera.Xpi[i][j] = 0;
			}
		}
	}
	m_camera.Xpi[2][2] = tan((m_camera.FOV * PI / 180) / 2);
	m_camera.Xpi[3][2] = tan((m_camera.FOV * PI / 180) / 2);

	//Push Matrix
	GzPushMatrix(Xsp);
	GzPushMatrix(m_camera.Xpi);
	GzPushMatrix(m_camera.Xiw);

	return GZ_SUCCESS;
}

int GzRender::GzPutCamera(GzCamera camera)
{
	/* HW 3.8
	/*- overwrite renderer camera structure with new camera definition
	*/
	m_camera = camera;
	return GZ_SUCCESS;
}


int GzRender::GzPushMatrix(GzMatrix	matrix)
{
	/*
	- push a matrix onto the Ximage stack
	- check for stack overflow
	*/
	if (matlevel >= MATLEVELS) {
		return GZ_FAILURE;
	}

	else {
		GzMatrix result;
		GzMatrix I;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				if (i == j) {
					I[i][j] = 1;
				}
				else
					I[i][j] = 0;
			}
		}
		if (matlevel == -1) {

			for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 4; j++) {
					result[i][j] = 0;
					for (int k = 0; k < 4; k++) {
						result[i][j] += (I[i][k] * matrix[k][j]);
					}
				}
			}
		} 
		else {
			for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 4; j++) {
					result[i][j] = 0;
					for (int k = 0; k < 4; k++) {
						result[i][j] += (Ximage[matlevel][i][k] * matrix[k][j]);
					}
				}
			}

		} 
		matlevel++;

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				Ximage[matlevel][i][j] = result[i][j];
			}
		}


		if (matlevel == 1 || matlevel == 0) {

			for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 4; j++) {
					Xnorm[matlevel][i][j] = I[i][j];
				}
			}
		} 
		else {
			GzMatrix  unitaryResult, tmp_xnorm;
			for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 4; j++) {
					unitaryResult[i][j] = matrix[i][j];
				}
			}
			unitaryResult[0][3] = 0;
			unitaryResult[1][3] = 0;
			unitaryResult[2][3] = 0;
			unitaryResult[3][3] = 1;

			for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 4; j++) {
					tmp_xnorm[i][j] = 0.0;
					for (int k = 0; k < 4; k++) {
						tmp_xnorm[i][j] = tmp_xnorm[i][j] + Xnorm[matlevel - 1][i][k] * unitaryResult[k][j];
					}
				}

			}
			float mod_norm = sqrt(pow(tmp_xnorm[0][0],2) + pow(tmp_xnorm[0][1], 2) + pow(tmp_xnorm[0][2],2));
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					tmp_xnorm[i][j] = tmp_xnorm[i][j] / mod_norm;
				}
			}

			for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 4; j++) {
					Xnorm[matlevel][i][j] = tmp_xnorm[i][j];
				}
			}
		}//else ends
	}
	return GZ_SUCCESS;
}


int GzRender::GzPopMatrix()
{
	/* HW 3.10
	- pop a matrix off the Ximage stack
	- check for stack underflow
	*/
	if (matlevel < 0) {
		return GZ_FAILURE;
	}
	else {
		matlevel--;
	}
	return GZ_SUCCESS;
}

int GzRender::GzPut(int i, int j, GzIntensity r, GzIntensity g, GzIntensity b, GzIntensity a, GzDepth z)
{
	/* HW1.4 write pixel values into the buffer */
	if (pixelbuffer == NULL)
	{
		return GZ_FAILURE;
	}
	if (i >= 0 && i < xres && j >= 0 && j < yres) {
		GzIntensity red = (min(r, 4095) < 0) ? 0 : min(r, 4095);
		GzIntensity green = (min(g, 4095) < 0) ? 0 : min(g, 4095);
		GzIntensity blue = (min(b, 4095) < 0) ? 0 : min(b, 4095);
		int idx = ARRAY(i, j);
		pixelbuffer[idx].red = red;
		pixelbuffer[idx].green = green;
		pixelbuffer[idx].blue = blue;
		pixelbuffer[idx].alpha = a;
		pixelbuffer[idx].z = z;
	}
	return GZ_SUCCESS;
}


int GzRender::GzGet(int i, int j, GzIntensity* r, GzIntensity* g, GzIntensity* b, GzIntensity* a, GzDepth* z)
{
	/* HW1.5 retrieve a pixel information from the pixel buffer */
	if (pixelbuffer == NULL)
	{
		return GZ_FAILURE;
	}

	if (i >= 0 && i < xres && j >= 0 && j < yres) {
		int idx = ARRAY(i, j);
		*r = pixelbuffer[idx].red;
		*g = pixelbuffer[idx].green;
		*b = pixelbuffer[idx].blue;
		*a = pixelbuffer[idx].alpha;
		*z = pixelbuffer[idx].z;
	}
	return GZ_SUCCESS;
}


int GzRender::GzFlushDisplay2File(FILE* outfile)
{
	/* HW1.6 write image to ppm file -- "P6 %d %d 255\r" */
	if (pixelbuffer == NULL) {
		return GZ_FAILURE;
	}

	fprintf(outfile, "P6 %d %d 255\n", xres, yres);

	for (int i = 0; i < xres; i++)
	{
		for (int j = 0; j < yres; j++)
		{
			GzPixel tmp = pixelbuffer[ARRAY(i, j)];
			char pred = tmp.red >> 4;
			char pgreen = tmp.green >> 4;
			char pblue = tmp.blue >> 4;
			fwrite(&pred, 1, 1, outfile);
			fwrite(&pgreen, 1, 1, outfile);
			fwrite(&pblue, 1, 1, outfile);
		}
	}

	return GZ_SUCCESS;
}

int GzRender::GzFlushDisplay2FrameBuffer()
{
	/* HW1.7 write pixels to framebuffer:
		- put the pixels into the frame buffer
		- CAUTION: when storing the pixels into the frame buffer, the order is blue, green, and red
		- NOT red, green, and blue !!!
	*/
	if (pixelbuffer == NULL) {
		return GZ_FAILURE;
	}

	if (framebuffer == NULL) {
		return GZ_FAILURE;
	}


	ApplyColorQuantization(4, this);
	EdgeDetection(this);
	//ApplyComicEffect(this);


	for (int i = 0; i < xres; i++)
	{
		for (int j = 0; j < yres; j++)
		{
			GzPixel tmp = pixelbuffer[ARRAY(i, j)];
			char pred = tmp.red >> 4;
			char pgreen = tmp.green >> 4;
			char pblue = tmp.blue >> 4;
			framebuffer[3 * ARRAY(i, j)] = pblue;
			framebuffer[(3 * ARRAY(i, j)) + 1] = pgreen;
			framebuffer[(3 * ARRAY(i, j)) + 2] = pred;
		}
	}

	return GZ_SUCCESS;
}


/***********************************************/
/* HW2 methods: implement from here */


int GzRender::GzPutAttribute(int numAttributes, GzToken* nameList, GzPointer* valueList)
{
	/* HW 2.1
	-- Set renderer attribute states (e.g.: GZ_RGB_COLOR default color)
	-- In later homeworks set shaders, interpolaters, texture maps, and lights
	*/
	for (int i = 0; i < numAttributes; i++) {
		if (nameList[i] == GZ_RGB_COLOR) {
			float* color = (float*)valueList[i];
			flatcolor[0] = color[0];
			flatcolor[1] = color[1];
			flatcolor[2] = color[2];
		}
		else if (nameList[i] == GZ_DIRECTIONAL_LIGHT) {
			GzLight* dLight = (GzLight*)valueList[i];
			for (int j = 0; j < 3; j++) {
				lights[numlights].direction[j] = dLight->direction[j];
				lights[numlights].color[j] = dLight->color[j];

			}
			numlights++;
		}
		else if (nameList[i] == GZ_AMBIENT_LIGHT) {
			GzLight* aLight = (GzLight*)valueList[i];
			ambientlight.direction[0] = aLight->direction[0];
			ambientlight.direction[1] = aLight->direction[1];
			ambientlight.direction[2] = aLight->direction[2];
			ambientlight.color[0] = aLight->color[0];
			ambientlight.color[1] = aLight->color[1];
			ambientlight.color[2] = aLight->color[2];
		}
		else if (nameList[i] == GZ_DIFFUSE_COEFFICIENT) {
			float* diffC = (float*)valueList[i];
			for (int j = 0; j < 3; j++) {
				Kd[j] = diffC[j];
			}
		}
		else if (nameList[i] == GZ_SPECULAR_COEFFICIENT) {
			float* specC = (float*)valueList[i];
			for (int j = 0; j < 3; j++) {
				Ks[j] = specC[j];
			}
		}
		else if (nameList[i] == GZ_AMBIENT_COEFFICIENT) {
			float* ambiC = (float*)valueList[i];
			for (int j = 0; j < 3; j++) {
				Ka[j] = ambiC[j];
			}
		}
		else if (nameList[i] == GZ_DISTRIBUTION_COEFFICIENT) {
			spec = *(float*)valueList[i];
		}
		else if (nameList[i] == GZ_INTERPOLATE) {
			interp_mode = *(int*)valueList[i];
		}
	}
	return GZ_SUCCESS;
}

int GzRender::GzPutTriangle(int	numParts, GzToken* nameList, GzPointer* valueList)
/* numParts - how many names and values */
{
	/* HW 2.2
	-- Pass in a triangle description with tokens and values corresponding to
		  GZ_NULL_TOKEN:		do nothing - no values
		  GZ_POSITION:		3 vert positions in model space
	-- Invoke the rastrizer/scanline framework
	-- Return error code
	*/
	if (nameList[0] == GZ_NULL_TOKEN) {
		return GZ_FAILURE;
	}
	GzCoord* position;
	GzCoord* normalList;

	for (int i = 0; i < numParts; i++) {
		if (nameList[i] == GZ_POSITION) {
			position = (GzCoord*)valueList[i];
		}
		if (nameList[i] == GZ_NORMALS)
		{
			normalList = (GzCoord*)valueList[i];
		}
	}//end

		GzCoord vertices[3], normals[3];
		//store 3 vertices in low to high Y ordering 

		//homogeneous coord
		float homogeneousVertices[3][4];
		float homogeneousNormal[3][4];

		float transedVertices[3][4];
		float transedNormal[3][4];

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				homogeneousVertices[i][j] = position[i][j];
				homogeneousNormal[i][j] = normalList[i][j];
			}
			homogeneousVertices[i][3] = 1;
			homogeneousNormal[i][3] = 1;
		}


		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 4; j++) {
				transedVertices[i][j] = 0;
				transedNormal[i][j] = 0;
				for (int k = 0; k < 4; k++) {
					transedVertices[i][j] += Ximage[matlevel][j][k] * homogeneousVertices[i][k];
					transedNormal[i][j] += Xnorm[matlevel][j][k] * homogeneousNormal[i][k];
				}
			}
		}

		//4d->3d
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				vertices[i][j] = transedVertices[i][j] / transedVertices[i][3];
				normals[i][j] = transedNormal[i][j] / transedNormal[i][3];
			}
		}

		//Skip negative z value triangle
		/*
				if (vertices[0][2] < 0 || vertices[1][2] < 0 || vertices[2][2] < 0) {
					return GZ_SUCCESS;
		}
		*/
		GzCoord face_N;
		face_N[0] = normals[0][0];
		face_N[1] = normals[0][1];
		face_N[2] = normals[0][2];


		if (vertices[0][1] > vertices[1][1]) {
			for (int i = 0; i < 3; i++) {
				std::swap(vertices[0][i], vertices[1][i]);
				std::swap(normals[0][i], normals[1][i]);
			}
		}

		if (vertices[0][1] > vertices[2][1]) {
			for (int i = 0; i < 3; i++) {
				std::swap(vertices[0][i], vertices[2][i]);
				std::swap(normals[0][i], normals[2][i]);
			}
		}

		if (vertices[1][1] > vertices[2][1]) {
			for (int i = 0; i < 3; i++) {
				std::swap(vertices[1][i], vertices[2][i]);
				std::swap(normals[1][i], normals[2][i]);
			}
		}



		//determine L/R TOP/BOT relationship

		//edge case1 two vers same Y horrizontal bot/top edge
		if (approximatelyEqual(vertices[0][1], vertices[1][1])) {//bottom edge
			if (vertices[0][0] > vertices[1][0]) {
				for (int i = 0; i < 3; i++) {
					std::swap(vertices[0][i], vertices[1][i]);
					std::swap(normals[0][i], normals[1][i]);
				}
			}
		}
		else if (approximatelyEqual(vertices[1][1], vertices[2][1])) {//top edge
			if (vertices[1][0] < vertices[2][0]) {
				for (int i = 0; i < 3; i++) {
					std::swap(vertices[1][i], vertices[2][i]);
					std::swap(normals[1][i], normals[2][i]);
				}
			}
		}
		else { //determine the CW order
			float midYX = vertices[1][0];
			float midYY = vertices[1][1];
			//slope = (Y3-Y1)/(X3-X1)
			float slope = (vertices[0][1] - vertices[2][1]) / (vertices[0][0] - vertices[2][0]);
			float midX = vertices[0][0] + (midYY - vertices[0][1]) / slope;
			if (midX > midYX) {
				for (int i = 0; i < 3; i++) {
					std::swap(vertices[1][i], vertices[2][i]);
					std::swap(normals[1][i], normals[2][i]);
				}
			}
		}

		//edges are correct sorted in CW, Z-interpolation
		float vec1[] = { vertices[1][0] - vertices[0][0], vertices[1][1] - vertices[0][1], vertices[1][2] - vertices[0][2] };
		float vec2[] = { vertices[2][0] - vertices[1][0], vertices[2][1] - vertices[1][1], vertices[2][2] - vertices[1][2] };

		float* crossP = crossProduct(vec1, vec2); //(A,B,C)
		//3D plane Equation crossP (A,B,C,D)
		float A, B, C, D;
		A = crossP[0];
		B = crossP[1];
		C = crossP[2];
		D = -1 * (A * vertices[0][0] + B * vertices[0][1] + C * vertices[0][2]);

		//bounding box
		int minX, maxX, minY, maxY;

		minX = min(min(vertices[0][0], vertices[1][0]), vertices[2][0]);
		maxX = max(max(vertices[0][0], vertices[1][0]), vertices[2][0]);
		minY = min(min(vertices[0][1], vertices[1][1]), vertices[2][1]);
		maxY = max(max(vertices[0][1], vertices[1][1]), vertices[2][1]);


////////Color for flat shading
		GzColor color_flat = { 0, 0, 0 };
		getcolor(this, face_N, color_flat);

///////Color for gouraud shading
		GzColor color_gouraud = { 0,0,0 };
		GzColor color_v1 = { 0,0,0 };
		GzColor color_v2 = { 0,0,0 };
		GzColor color_v3 = { 0,0,0 };

		getcolor(this, normals[0], color_v1);
		getcolor(this, normals[1], color_v2);
		getcolor(this, normals[2], color_v3);

		//Interpolating RED
		float red_vec1[] = { vertices[1][0] - vertices[0][0], vertices[1][1] - vertices[0][1], color_v2[0] - color_v1[0]};
		float red_vec2[] = { vertices[2][0] - vertices[1][0], vertices[2][1] - vertices[1][1], color_v3[0] - color_v2[0]};
		float* redCrossP = crossProduct(red_vec1, red_vec2); //(A,B,C)
		//3D plane Equation crossP (A,B,C,D)
		float redA, redB, redC, redD;
		redA = redCrossP[0];
		redB = redCrossP[1];
		redC = redCrossP[2];
		redD = -1 * (redA * vertices[0][0] + redB * vertices[0][1] + redC * color_v1[0]);
			
		//Interpolating GREEN
		float green_vec1[] = { vertices[1][0] - vertices[0][0], vertices[1][1] - vertices[0][1], color_v2[1] - color_v1[1] };
		float green_vec2[] = { vertices[2][0] - vertices[1][0], vertices[2][1] - vertices[1][1], color_v3[1] - color_v2[1] };
		float* greenCrossP = crossProduct(green_vec1, green_vec2); //(A,B,C)
		//3D plane Equation crossP (A,B,C,D)
		float greenA, greenB, greenC, greenD;
		greenA = greenCrossP[0];
		greenB = greenCrossP[1];
		greenC = greenCrossP[2];
		greenD = -1 * (greenA * vertices[0][0] + greenB * vertices[0][1] + greenC * color_v1[1]);

		//Interpolating Blue
		float blue_vec1[] = { vertices[1][0] - vertices[0][0], vertices[1][1] - vertices[0][1], color_v2[2] - color_v1[2] };
		float blue_vec2[] = { vertices[2][0] - vertices[1][0], vertices[2][1] - vertices[1][1], color_v3[2] - color_v2[2] };
		float* blueCrossP = crossProduct(blue_vec1, blue_vec2); //(A,B,C)
		//3D plane Equation crossP (A,B,C,D)
		float blueA, blueB, blueC, blueD;
		blueA = blueCrossP[0];
		blueB = blueCrossP[1];
		blueC = blueCrossP[2];
		blueD = -1 * (blueA * vertices[0][0] + blueB * vertices[0][1] + blueC * color_v1[2]);

///////Color for Phong shading
		//Interpolating normalX
		float normalX_vec1[] = { vertices[1][0] - vertices[0][0], vertices[1][1] - vertices[0][1], normals[1][0] - normals[0][0]};
		float normalX_vec2[] = { vertices[2][0] - vertices[1][0], vertices[2][1] - vertices[1][1], normals[2][0] - normals[1][0]};
		float* normalXCrossP = crossProduct(normalX_vec1, normalX_vec2); //(A,B,C)
		//3D plane Equation crossP (A,B,C,D)
		float normalXA, normalXB, normalXC, normalXD;
		normalXA = normalXCrossP[0];
		normalXB = normalXCrossP[1];
		normalXC = normalXCrossP[2];
		normalXD = -1 * (normalXA * vertices[0][0] + normalXB * vertices[0][1] + normalXC * normals[0][0]);

		//Interpolating normalY
		float normalY_vec1[] = { vertices[1][0] - vertices[0][0], vertices[1][1] - vertices[0][1], normals[1][1] - normals[0][1] };
		float normalY_vec2[] = { vertices[2][0] - vertices[1][0], vertices[2][1] - vertices[1][1], normals[2][1] - normals[1][1] };
		float* normalYCrossP = crossProduct(normalY_vec1, normalY_vec2); //(A,B,C)
		//3D plane Equation crossP (A,B,C,D)
		float normalYA, normalYB, normalYC, normalYD;
		normalYA = normalYCrossP[0];
		normalYB = normalYCrossP[1];
		normalYC = normalYCrossP[2];
		normalYD = -1 * (normalYA * vertices[0][0] + normalYB * vertices[0][1] + normalYC * normals[0][1]);

		//Interpolating normalZ
		float normalZ_vec1[] = { vertices[1][0] - vertices[0][0], vertices[1][1] - vertices[0][1], normals[1][2] - normals[0][2] };
		float normalZ_vec2[] = { vertices[2][0] - vertices[1][0], vertices[2][1] - vertices[1][1], normals[2][2] - normals[1][2] };
		float* normalZCrossP = crossProduct(normalZ_vec1, normalZ_vec2); //(A,B,C)
		//3D plane Equation crossP (A,B,C,D)
		float normalZA, normalZB, normalZC, normalZD;
		normalZA = normalZCrossP[0];
		normalZB = normalZCrossP[1];
		normalZC = normalZCrossP[2];
		normalZD = -1 * (normalZA * vertices[0][0] + normalZB * vertices[0][1] + normalZC * normals[0][2]);




		for (int i = minX; i <= maxX; i++) {
			for (int j = minY; j <= maxY; j++) {
				float res1, res2, res3, dY, dX;
				//dYx + (-dXy) + (dXY - dYX) = 0
				res1 = (vertices[1][1] - vertices[0][1]) * (float)i
					- (vertices[1][0] - vertices[0][0]) * (float)j
					+ ((vertices[1][0] - vertices[0][0]) * vertices[0][1] - (vertices[1][1] - vertices[0][1]) * vertices[0][0]);

				res2 = (vertices[2][1] - vertices[1][1]) * (float)i
					- (vertices[2][0] - vertices[1][0]) * (float)j
					+ ((vertices[2][0] - vertices[1][0]) * vertices[1][1] - (vertices[2][1] - vertices[1][1]) * vertices[1][0]);

				res3 = (vertices[0][1] - vertices[2][1]) * (float)i
					- (vertices[0][0] - vertices[2][0]) * (float)j
					+ ((vertices[0][0] - vertices[2][0]) * vertices[2][1] - (vertices[0][1] - vertices[2][1]) * vertices[2][0]);;

				if ((res1 > 0 && res2 > 0 && res3 > 0) || (res1 < 0 && res2 < 0 && res3 < 0 || (res1 == 0 && res2 == 0 && res3 == 0))) {
					int zInter = (int)((-1 * (A * (float)i + B * (float)j + D)) / C + 0.5);
					GzIntensity r, g, b, a;
					GzDepth z;
					GzGet(i, j, &r, &g, &b, &a, &z);
					if (z > zInter) {
						if (interp_mode == GZ_FLAT) {
							GzPut(i, j, ctoi(color_flat[0]), ctoi(color_flat[1]), ctoi(color_flat[2]), 1, zInter);
						}
						else if (interp_mode == GZ_COLOR) {
							color_gouraud[0] = -1 * (redA * i + redB * j + redD) / redC;
							color_gouraud[1] = -1 * (greenA * i + greenB * j + greenD) / greenC;
							color_gouraud[2] = -1 * (blueA * i + blueB * j + blueD) / blueC;
							GzPut(i, j, ctoi(color_gouraud[0]), ctoi(color_gouraud[1]), ctoi(color_gouraud[2]), 1, zInter);
						}
						else if (interp_mode == GZ_NORMALS) {
							GzColor color_phong = {0,0,0};
							GzCoord pixel_normal;
							pixel_normal[0] = -1 * (normalXA * i + normalXB * j + normalXD) / normalXC;
							pixel_normal[1] = -1 * (normalYA * i + normalYB * j + normalYD) / normalYC;
							pixel_normal[2] = -1 * (normalZA * i + normalZB * j + normalZD) / normalZC;
							getcolor(this,pixel_normal,color_phong);
							GzPut(i, j, ctoi(color_phong[0]), ctoi(color_phong[1]), ctoi(color_phong[2]), 1, zInter);
						}
					}
				}
			}

		}
	return GZ_SUCCESS;
}

