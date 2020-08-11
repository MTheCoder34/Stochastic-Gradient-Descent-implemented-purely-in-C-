//Written by Milan Neducza, All Rights Reserved

#include <iostream>
#include <stdio.h> 
#include <conio.h> 
#include <algorithm>
#include <string>
#define E 2.718f


using namespace std;

void Init(int size_n, float* weights, int w_length_x, int w_length_y, float* bias, int bias_length) {
	int k = 0;
	int g = 0;
	srand(1000);
	for (int i = 1; i < size_n; i++)
	{
		while(k < i * w_length_x * w_length_y)
		{
			*(weights + k) = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
			k++;
		}
		for (int g = 0; g < i * bias_length;g++)
		{
			*(bias + g) = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		}
	}
	delete &k;
	delete &g;
}
void MatrixMultiplication(float* a, int a_size,float** b, int b_x_size, int b_y_size, float* c) {
	for (int i = 0; i < b_y_size; i++)
	{
		c[i] = 0;
	}
	for (int k = 0; k < b_y_size; k++)
	{
		for (int i = 0; i < b_x_size; i++)
		{
			c[k] += a[i] * b[i][k];
		}
	}
}
void MatrixAddition(float* a, int a_size, float* b, int b_size, float* c) {
	for (int i = 0; i < a_size; i++)
	{
		c[i] = 0;
	}
	for (int i = 0; i < a_size; i++)
	{
		c[i] = a[i] + b[i];
	}
}
float Sigmoid(float x) {
	return 1 / (1 + pow(E, -x));
}
void SigmoidFunction(float* a, int a_len, float* b) {
	for (int i = 0; i < a_len; i++)
	{
		b[i] = Sigmoid(a[i]);
	}
}
void Transform(float*** a, float*** z, int batch_index, int* layers, int num_of_layers, float*** w, float** b) {
	for (int i = 1; i < num_of_layers; i++)
	{
		float* c = new float[layers[i - 1]];
		MatrixMultiplication(a[batch_index][i - 1], layers[i - 1], w[i - 1], layers[i], layers[i - 1],c);
		float* d = new float[layers[i - 1]];
		MatrixAddition(c, layers[i], b[i - 1], layers[i], d);
		//cout << "Prev B:" << z[batch_index][i][0] << endl;
		*z[batch_index][i] = *d;//itt voltál
		//cout << "B:" << z[batch_index][i][0] << endl;
		SigmoidFunction(d,layers[i], a[batch_index][i]);
		delete[] c;
		delete[] d;
	}
}
float Cost(float a, float b) {
	return pow(a - b, 2);
}
float SigmoidPrime(float x) {
	return Sigmoid(x) * (1 - Sigmoid(x));
}
void CostFunction(float** Error, int a_len, float** y, float*** a, int num_of_layers, int batch_index) {
	for (int i = 0; i < a_len; i++)
	{
		Error[batch_index][i] = Cost(a[batch_index][num_of_layers-1][i], y[batch_index][i]);
	}
}

void CountOverallCost(float*** a, int num_of_minibatches, int num_of_layers, int a_len, float** y, float* ptr_loc, float **Error) {
	*ptr_loc = 0;
	for (int i = 0; i < num_of_minibatches; i++)
	{
		for (int k = 0; k < a_len; k++)
		{
			Error[i][k] = pow(a[i][num_of_layers - 1][k] - y[i][k], 2);
			*ptr_loc += Error[i][k];
		}
	}
}
void SetWeightsAndBiases(float learning_rate,float* a, float* z, float* DelErrorPerDelH, float*** w, int k, int w_x, int w_y, float** b)
{
	for (int i = 0; i < w_x; i++)
	{
		for (int j = 0; j < w_y; j++)
		{
			w[k-1][i][j] = w[k-1][i][j] - (learning_rate * (a[j] * SigmoidPrime(z[i]) * DelErrorPerDelH[i]));
		}
		b[k-1][i] = b[k-1][i] - learning_rate * (SigmoidPrime(z[i]) * DelErrorPerDelH[i]);
	}
}
void DelErrorPerDelA(float *h, float **w, int w_x, int w_y, float* DelErrorPerDelH)
{
	float* output = new float[w_y];
	for (int i = 0; i < w_y; i++)
	{
		output[i] = 0;
	}
	for (int i = 0; i < w_y; i++)
	{
		for (int q = 0; q < w_x; q++)
		{
			/*cout <<"H: " <<h[q] << endl;
			cout << "W: " << w[q][i] << endl;
			cout << "DelErrorPerDelA: "<< DelErrorPerDelH[q] << endl;
			//cout << "Derivative:" << SigmoidPrime(h[q]) * w[q][i] * DelErrorPerDelH[q] << endl;
			cout << "Output1: " << output[i] << endl;*/
			output[i] = output[i] + SigmoidPrime(h[q]) * w[q][i] * DelErrorPerDelH[q];
			//cout << "Output: " << output[i] << endl;
		}
	}
	memcpy(DelErrorPerDelH, output, sizeof output);
	delete[] output;
}
void Fit(float** Error1, float*** w, float** b, float*** a, int num_of_batches, int* net_layers, int num_of_layers, float*** z, float** y)
{
	float lrnr = 0.1;
	float OverallCost = 0;

	for (int i = 0; i < num_of_batches; i++)
	{
		//cout << a[0][1][0] << endl;
		Transform(a, z, i, net_layers, num_of_layers, w, b);
		//cout << a[0][1][0] << endl;
	}
	CountOverallCost(a, num_of_batches, num_of_layers, net_layers[num_of_layers-1], y, &OverallCost,Error1);
	int i = 0;
	int n = 0;
	while (OverallCost > 0.001)
	{
		
		for (i = 0; i < num_of_batches; i++)
		{
			Transform(a, z, i, net_layers, num_of_layers, w, b);
			CostFunction(Error1, net_layers[num_of_layers - 1], y, a, num_of_layers, i);
			float* ErrorPerA = new float[net_layers[num_of_layers - 1]];
			for (int k = 0; k < net_layers[num_of_layers - 1]; k++)
			{
				ErrorPerA[k] = 2 * (a[i][num_of_layers - 1][k] - y[i][k]);
			}
			for (int k = num_of_layers - 2; k > 0; k--)
			{
				SetWeightsAndBiases(lrnr, a[i][k], z[i][k + 1], ErrorPerA, w, k, net_layers[k+1], net_layers[k], b);
				//cout << ErrorPerA[0] << endl;
				DelErrorPerDelA(z[i][k + 1], w[k], net_layers[k+1], net_layers[k], ErrorPerA);
				//cout << ErrorPerA[0] << endl;
			}
			delete[] (ErrorPerA);
		}
		for (int i = 0; i < num_of_batches; i++)
		{
			//cout << a[0][1][0] << endl;
			Transform(a, z, i, net_layers, num_of_layers, w, b);
			//cout << a[0][1][0] << endl;
		}
		CountOverallCost(a, num_of_batches, num_of_layers, net_layers[num_of_layers - 1], y, &OverallCost, Error1);
		n++;
		if(n % 10 == 0) std::cout << OverallCost << endl;
	}
	cout << OverallCost << endl;
	delete[] Error1;
}
int main()
{
	int layers[] = { 2,2,1 };
	int num_of_batches = 4;
	float*** w = new float** [2];
	float** b = new float* [2];
	for (int i = 1; i < 3; i++)
	{
		float** w_sub = new float* [layers[i]];
		float* b_sub = new float[layers[i]];
		for (int k = 0; k < layers[i]; k++)
		{
			float* w_sub1 = new float[layers[i - 1]];
			for (int j = 0; j < layers[i - 1]; j++)
			{
				w_sub1[j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
			}
			w_sub[k] = w_sub1;
			b_sub[k] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		}
		b[i - 1] = b_sub;
		w[i - 1] = w_sub;
	}

	float* input1 = new float[2]{ 0, 0 };
	float* input2 = new float[2]{ 0, 1 };
	float* input3 = new float[2]{ 1, 0 };
	float* input4 = new float[2]{ 1, 1 };

	float** input = new float* [4]{ input1, input2, input3, input4};//These are the training samples
	float*** z = new float** [num_of_batches];
	float*** a = new float** [num_of_batches];
	for (int i = 0; i < num_of_batches; i++)
	{
		a[i] = new float* [sizeof layers / sizeof(int)];
		z[i] = new float* [sizeof layers / sizeof(int)];
		a[i][0] = input[i];
		z[i][0] = input[i];
		for (int k = 1; k < sizeof layers/ sizeof (int); k++)
		{
			float* sub_a = new float[layers[k]];
			a[i][k] = sub_a;
			z[i][k] = sub_a;
		}
	}
	float** y = new float*[1];//These are the required values


	float* y_ = new float[1];
	y_[0] = 0;

	float* y_1 = new float[1];
	y_1[0] = 1;

	float* y_2 = new float[1];
	y_2[0] = 1;

	float* y_3 = new float[1];
	y_3[0] = 1;


	y[0] = y_;
	y[1] = y_1;
	y[2] = y_2;
	y[3] = y_3;
	float OverallCost = 0;
	float** error1 = new float* [num_of_batches];
	for (int i = 0; i < num_of_batches; i++)
	{
		float* a___ = new float[layers[2]];
		for (int k = 0; k < layers[2]; k++)
		{
			a___[k] = 0;
		}
		error1[i] = a___;
	}
	Fit(error1,w, b, a, num_of_batches, layers, 3, z, y);//We train the network here
	printf("%c", _getch());
}
