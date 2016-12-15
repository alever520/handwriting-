#include<iostream>
#include<opencv2\opencv.hpp>
#include<string>
#include<vector>
using namespace cv;
using namespace std;
void resizeImage(Mat& srcImage, Size size);
void binaryInit(Mat& srcDataMat, Size size);
void binaryInit(Mat& srcDataMat, Size size);
void normal(Mat& srcImage, Mat& dstImage);
void preProcess(const Mat& srcImage, Mat& dstImage);



//����ͼ���С

void resizeImage(Mat& srcImage, Size size)

{

	if (srcImage.rows * srcImage.cols < size.area())

		resize(srcImage, srcImage, size, 0, 0, INTER_CUBIC);

	else if (srcImage.rows * srcImage.cols > size.area())

		resize(srcImage, srcImage, size, 0, 0, INTER_AREA);

}



//ͼ�� ��ֵ��ʼ��

void binaryInit(Mat& srcDataMat, Size size)

{
	
	srcDataMat.convertTo(srcDataMat, CV_8UC1);

	normal(srcDataMat, srcDataMat);

	//resizeImage(srcDataMat, size);

	threshold(srcDataMat, srcDataMat, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);//��ֵ��Ҫ��ͼ��ΪCV_8UC1

	//srcDataMat.convertTo(srcDataMat, CV_32FC1);//����ѧϰ�㷨����������ΪCV_32FC1

}



//ͼ���һ�� ����

void normal(Mat& srcImage, Mat& dstImage)

{

	//�ҵ���Ч����roi�����ַ�����Ӿ���



	int bottom = srcImage.rows + 1;

	int top = 0;

	int left = srcImage.cols + 1;

	int right = 0;

	for (int i = 0; i < srcImage.rows; ++i)

	{

		for (int j = 0; j < srcImage.cols; ++j)

		{

			if (srcImage.at<uchar>(i, j) > 0)

			{

				bottom = min(bottom, i);

				top = max(top, i);

				left = min(left, j);

				right = max(right, j);

			}

		}

	}

	Rect rec = Rect(left, bottom, right - left + 1, top - bottom + 1);

	Mat roi = srcImage(rec);



	//ͼƬ���в���һ��

	int width = roi.cols;

	int height = roi.rows;

	int longLen = max(width, height);

	int shortLen = width + height - longLen;
	if (roi.type() != CV_8UC1)

	{

		cvtColor(roi, roi, CV_BGR2GRAY);

	}


	dstImage = Mat(longLen, longLen, CV_8UC1, Scalar::all(0));

	roi.copyTo(dstImage(Rect((longLen - width) / 2, (longLen - height) / 2, width, height)));

	//imshow("shit2", dstImage);

	//waitKey(1000);

}



//Ԥ���� �˲� ��ȡ��Ե ��ʴ���Ӷϵ�

void preProcess(const Mat& srcImage, Mat& dstImage)

{

	Mat tmpImage = srcImage.clone();

	if (tmpImage.type() != CV_8UC1)

	{

		cvtColor(tmpImage, tmpImage, CV_BGR2GRAY);

	}

	//��ֵ�˲�Ԥ�������

	//GaussianBlur(tmpImage, tmpImage, Size(3, 3), 0, 0, BORDER_DEFAULT);

	medianBlur(tmpImage, tmpImage, 3);

	//��̬ѧ�˲� ������Ԥ����

	Mat element = getStructuringElement(MORPH_RECT, Size(7, 7));

	//morphologyEx(tmpImage, tmpImage, MORPH_OPEN, element);



	//imshow("��˹�˲��ӿ�����", tmpImage);

	//waitKey(500);



	//canny��ȡ����

	Canny(tmpImage, dstImage, 75, 100, 3);

	//imshow("canny������ȡ", dstImage);

	//waitKey(500);



	//��̬ѧ�˲����������մ���

	element = getStructuringElement(MORPH_RECT, Size(3, 3));

	morphologyEx(dstImage, dstImage, MORPH_DILATE, element);

	//imshow("��̬ѧ����", tmpImage4);

	//waitKey(500);



	//��ֵ�˲����������

	//medianBlur(tmpImage4, dstImage, 3);

	//imshow("��ֵ�˲���", dstImage);

	//waitKey(500);

}



//���մ��ϵ��� �����ҵ��Ķ�˳������

bool cmp(const Rect& a, const Rect& b)

{

	return a.x < b.x;

}

bool cmp2(const Rect& a, const Rect& b)

{

	return a.tl().y < b.tl().y;

}

void sortRect(vector<Rect>& arr)

{

	sort(arr.begin(), arr.end(), cmp2);

	vector<Rect>::iterator s = arr.begin();

	vector<Rect>::iterator e = arr.end();

	vector<Rect>::iterator ptr = arr.begin();

	vector<Rect>::iterator preptr = ptr++;

	for (; ; ++ptr, ++preptr)

	{

		if (ptr == arr.end() || ptr->tl().y > preptr->br().y)

		{

			e = ptr;

			sort(s, e, cmp);

			s = ptr;

			if (ptr == arr.end())

				break;

		}

	}

}



//��ȡ����ĵ�����ĸ������

void getSegment(const Mat& srcImage, vector<Mat>& arr, Mat& showImage)

{

	Mat tmpImage = srcImage.clone();

	if (tmpImage.type() != CV_8UC1)

	{

		cvtColor(tmpImage, tmpImage, CV_BGR2GRAY);

	}

	threshold(tmpImage, tmpImage, 0, 255, CV_THRESH_BINARY);

	vector<vector<Point>> contours;

	findContours(tmpImage, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);//�˺������޸�tmpImage



	Mat tmpImage3 = srcImage.clone();

	vector<Rect> rectArr;

	for (int i = 0; i < contours.size(); ++i)

	{

		Rect rec = boundingRect(contours[i]);//��Ӿ���

		if (rec.area() >= 8 * 8)

		{

			rectArr.push_back(rec);

		}

	}



	sortRect(rectArr);//�Ծ��ΰ��Ķ���ʽ��������

					  cout << rectArr.size() << endl;

	for (int i = 0; i < rectArr.size(); ++i)

	{

		Mat tmp = tmpImage3(rectArr[i]);

		rectangle(showImage, rectArr[i], Scalar(100), 2);//������Ӿ���

		normal(tmp, tmp);

		arr.push_back(tmp);

	}

}



void preproduct()
{
	Mat src, dst;
	src = imread("1.png");
	imshow("2", src);
	Mat temp = src.clone();
	GaussianBlur(temp, temp, Size(3, 3), 0, 0);
	cvtColor(temp, temp, CV_BGR2GRAY);
	imshow("3", temp);
	Sobel(temp, temp, -1, 1, 1);
	//cvtColor(temp, temp, CV_BGR2HSV);
	threshold(temp, temp, 0, 255, CV_THRESH_OTSU);
	//Mat kernel = getStructuringElement(MORPH_CROSS, Size(3, 3));
	//morphologyEx(temp, temp, MORPH_CLOSE, kernel);
	//erode(temp, temp, kernel);
	//threshold(temp, temp, 5, 255, CV_THRESH_BINARY);
	//Mat temp2 = Mat::zeros(src.size(), src.type());
	//temp.convertTo(temp2, -1, 3, 0);
	//GaussianBlur(temp, temp, Size(3, 3), 0, 0);
	imshow("1", temp);
	waitKey(20161120);
}


// ������ɫ��ð�ɫ�����Ĳ���
void getwhite( Mat& srcImage, Mat& dstImage)
{
	dstImage = srcImage.clone();
	cvtColor(dstImage, dstImage, CV_BGR2GRAY);
	vector<Mat> channels;
	split(srcImage, channels);
	//imshow("1", channels.at(0));
	//imshow("2", channels.at(1));
	//imshow("3", channels.at(2));
	Mat b_Im, g_Im, r_Im;
	b_Im = channels.at(0);
	g_Im = channels.at(1);
	r_Im = channels.at(2);
	int nc = srcImage.cols;
	int nr = srcImage.rows;
	for (int i = 0; i < nr; i++)
	{
		for (int j = 0; j < nc; j++)
		{
			//  Ѱ������ɫͨ��������ֵ������120������ ȥ���󲿷ִ�����ɫ�����ص�
			if (b_Im.at<uchar>(i, j) > 120 && g_Im.at<uchar>(i, j) > 120 && r_Im.at<uchar>(i, j) > 120)
			{
				dstImage.at<uchar>(i, j) = 255;
			}
			else
				dstImage.at<uchar>(i, j) = 0;
		}
	}
	imshow("2", dstImage);
}


// ���ݰ�ɫ����ռ�ݵ����ȥ�����������������Ĳ���  ���н���
void cleanbyrow(Mat& srcImage, Mat& dstImage)
{
	dstImage = srcImage.clone();
	int nr = srcImage.rows;
	int nc = srcImage.cols;
	vector<int> count_row(nr, 0);
	for (int i = 0; i < nr; i++)
	{
		for (int j = 0; j < nc; j++)
			if (dstImage.at<uchar>(i, j) != 0) // ����ÿ�а�ɫ���ظ���
				count_row.at(i)++;
		if (count_row.at(i) < (int)nc / 4)  //ÿ���а�ɫ���������������ص�4��֮1
			for (int j = 0; j < nc; j++)
				dstImage.at<uchar>(i, j) = 0;
	}
		
}


//���ݰ�ɫ����ռ�ݵ����ȥ�����������������Ĳ���   ���н���
void cleanbycol(Mat& srcImage, Mat& dstImage)
{
	dstImage = srcImage.clone();
	int nr = srcImage.rows;
	int nc = srcImage.cols;
	vector<int> count_col(nc, 0);
	for (int j = 0; j < nc; j++)
	{
		for (int i = 0; i < nr; i++)
				if (dstImage.at<uchar>(i, j) != 0)  // ����ÿ���еİ�ɫ����
					count_col.at(j)++;
		if (count_col.at(j) < 10)  // ÿ���а�ɫ��������10��
			for (int i = 0; i < nr; i++)
				dstImage.at<uchar>(i, j) = 0;
	}
		
			
				
}


// �����Ӿ���
void getrect(Mat& srcImage,Mat& dstRect)
{
	Mat tmpImage = srcImage.clone();
	int nr = srcImage.rows;
	int nc = srcImage.cols;
	int bottom = srcImage.rows + 1;

	int top = 0;

	int left = srcImage.cols + 1;

	int right = 0;

	for (int i = 0; i < srcImage.rows; ++i)

	{

		for (int j = 0; j < srcImage.cols; ++j)

		{

			if (srcImage.at<uchar>(i, j) > 0)

			{

				bottom = min(bottom, i);

				top = max(top, i);

				left = min(left, j);

				right = max(right, j);

			}

		}

	}

	Rect rec = Rect(left, bottom, right - left + 1, top - bottom + 1);

	dstRect = srcImage(rec);

}
void main()
{
	Mat src, dst;
	vector<Mat> arr;
	//Size size = Size(2500, 2500);
	src = imread("1.png");
	imshow("1", src);
	dst = src.clone();
	getwhite(src, dst);
	cleanbyrow(dst, dst);
	cleanbycol(dst, dst);
	cleanbyrow(dst, dst);
	Mat rect;
	getrect(dst, rect);
	imshow("out", dst);
	imshow("4", rect);
	//preProcess(src, src); // �˲� ��ȡ��Ե ��ʴ����
	//binaryInit(src, size);// ��ֵ��
	//getSegment(src, arr, dst);
	//imshow("1", arr[2]);
	waitKey(20161120);
}