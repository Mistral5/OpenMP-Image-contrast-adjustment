#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

struct picture {
	unsigned char* bitmap;
	unsigned int type;
	unsigned int height;
	unsigned int width;
	unsigned int size;
	unsigned int maximumGrayValue;
};

char picSizeFinding(FILE* inputFile, struct picture* pic)
{
	fseek(inputFile, sizeof(char), SEEK_SET);

	if (fscanf(inputFile, "%u\n", &pic->type) < 1)
		return 1;

	if (fscanf(inputFile, "%u %u\n", &pic->width, &pic->height) < 1)
		return 1;

	if (fscanf(inputFile, "%u", &pic->maximumGrayValue) < 1)
		return 1;

	fseek(inputFile, sizeof(char), SEEK_CUR);

	return 0;
}

char readFile(FILE* inputFile, unsigned char* bitmap, unsigned int picSize)
{
	if (fread(bitmap, sizeof(char), picSize, inputFile) != picSize)
		return 1;

	return 0;
}

char writeFile(FILE* outputFile, struct picture* pic)
{
	if (fprintf(outputFile, "P%u\n", pic->type) < 0)
		return 1;

	if (fprintf(outputFile, "%u %u\n", pic->width, pic->height) < 0)
		return 1;

	if (fprintf(outputFile, "%u\n", pic->maximumGrayValue) < 0)
		return 1;

	if (fwrite(pic->bitmap, 1, sizeof(char) * pic->size, outputFile) != pic->size)
		return 1;

	return 0;
}

unsigned char minElFinding(unsigned int* histogram, int ignoredPixelQuantity)
{
	for (int i = 0; i < 256; i++)
	{
		ignoredPixelQuantity -= histogram[i];
		if (ignoredPixelQuantity < 0)
			return i;
	}
	return 255;
}

unsigned char maxElFinding(unsigned int* histogram, int ignoredPixelQuantity)
{
	for (int i = 255; i > -1; i--)
	{
		ignoredPixelQuantity -= histogram[i];
		if (ignoredPixelQuantity < 0)
			return i;
	}
	return 0;
}

char greyContrastCorrection(struct picture* pic, float ignoreRate)
{
	int ignoredPixelQuantity = pic->size * ignoreRate;
	unsigned int bitDepth = 256;

	unsigned int* histogram = (unsigned int*)calloc(bitDepth, sizeof(int));

	for (unsigned int i = 0; i < pic->size; i++)
		histogram[pic->bitmap[i]]++;

	unsigned char minElValue = minElFinding(histogram, ignoredPixelQuantity);
	unsigned char maxElValue = maxElFinding(histogram, ignoredPixelQuantity);

	free(histogram);

	if (minElValue == maxElValue)
		return 0;

	if (minElValue == 255 || maxElValue == 0)
		return 1;

	if (minElValue == 0 && maxElValue == 255)
		return 0;

	int maxMinDelta = maxElValue - minElValue;

	for (unsigned int i = 0; i < pic->size; i++)
	{
		if (pic->bitmap[i] <= minElValue)
		{
			pic->bitmap[i] = 0;
			continue;
		}

		if (pic->bitmap[i] >= maxElValue)
		{
			pic->bitmap[i] = 255;
			continue;
		}

		pic->bitmap[i] = ((pic->bitmap[i] - minElValue) * 255 / maxMinDelta);
	}

	return 0;
}

char anyContrastCorrection(struct picture* pic, float ignoreRate)
{
	int ignoredPixelQuantity = pic->height * pic->width * ignoreRate;
	unsigned int bitDepth = 256;

	unsigned int* histogramR = (unsigned int*)calloc(bitDepth, sizeof(int));
	unsigned int* histogramG = (unsigned int*)calloc(bitDepth, sizeof(int));
	unsigned int* histogramB = (unsigned int*)calloc(bitDepth, sizeof(int));

	for (unsigned int i = 0; i < pic->size; i += 3)
	{
		histogramR[pic->bitmap[i]]++;
		histogramG[pic->bitmap[i + 1]]++;
		histogramB[pic->bitmap[i + 2]]++;
	}

	unsigned char minElValue = minElFinding(histogramR, ignoredPixelQuantity);
	unsigned char maxElValue = maxElFinding(histogramR, ignoredPixelQuantity);
	unsigned char minElValueG = minElFinding(histogramG, ignoredPixelQuantity);
	unsigned char maxElValueG = maxElFinding(histogramG, ignoredPixelQuantity);
	unsigned char minElValueB = minElFinding(histogramB, ignoredPixelQuantity);
	unsigned char maxElValueB = maxElFinding(histogramB, ignoredPixelQuantity);

	if (minElValueG < minElValue)
		minElValue = minElValueG;
	if (minElValueB < minElValue)
		minElValue = minElValueB;

	if (maxElValueG > maxElValue)
		maxElValue = maxElValueG;
	if (maxElValueB > maxElValue)
		maxElValue = maxElValueB;

	free(histogramR);
	free(histogramG);
	free(histogramB);

	if (minElValue == maxElValue)
		return 0;

	if (minElValue == 255 || maxElValue == 0)
		return 1;

	if (minElValue == 0 && maxElValue == 255)
		return 0;

	int maxMinDelta = maxElValue - minElValue;

	for (unsigned int i = 0; i < pic->size; i++)
	{
		if (pic->bitmap[i] <= minElValue)
		{
			pic->bitmap[i] = 0;
			continue;
		}

		if (pic->bitmap[i] >= maxElValue)
		{
			pic->bitmap[i] = 255;
			continue;
		}

		pic->bitmap[i] = ((pic->bitmap[i] - minElValue) * 255 / maxMinDelta);
	}

	return 0;
}

char greyParallelContrastCorrection(struct picture* pic, float ignoreRate)
{
	int ignoredPixelQuantity = pic->size * ignoreRate;
	unsigned int bitDepth = 256;

	unsigned int* histogram = (unsigned int*)calloc(bitDepth, sizeof(int));

#pragma omp parallel
	{
		unsigned int* histogram_d = (unsigned int*)calloc(bitDepth, sizeof(int));

#pragma omp for schedule(static)
		for (unsigned int i = 0; i < pic->size; i++)
			histogram_d[pic->bitmap[i]]++;

		for (unsigned int i = 0; i < bitDepth; i++)
		{
#pragma omp atomic
			histogram[i] += histogram_d[i];
		}

		free(histogram_d);
	}

	unsigned char minElValue = minElFinding(histogram, ignoredPixelQuantity);
	unsigned char maxElValue = maxElFinding(histogram, ignoredPixelQuantity);

	free(histogram);

	if (minElValue == maxElValue)
		return 0;

	if (minElValue == 255 || maxElValue == 0)
		return 1;

	if (minElValue == 0 && maxElValue == 255)
		return 0;

	int maxMinDelta = maxElValue - minElValue;

#pragma omp parallel
	{
#pragma omp for schedule(static)
		for (unsigned int i = 0; i < pic->size; i++)
		{
			if (pic->bitmap[i] <= minElValue)
			{
				pic->bitmap[i] = 0;
				continue;
			}

			if (pic->bitmap[i] >= maxElValue)
			{
				pic->bitmap[i] = 255;
				continue;
			}

			pic->bitmap[i] = ((pic->bitmap[i] - minElValue) * 255 / maxMinDelta);
		}
	}

	return 0;
}

char anyParallelContrastCorrection(struct picture* pic, float ignoreRate)
{
	int ignoredPixelQuantity = pic->height * pic->width * ignoreRate;
	unsigned int bitDepth = 256;

	unsigned int* histogramR = (unsigned int*)calloc(bitDepth, sizeof(int));
	unsigned int* histogramG = (unsigned int*)calloc(bitDepth, sizeof(int));
	unsigned int* histogramB = (unsigned int*)calloc(bitDepth, sizeof(int));

#pragma omp parallel
	{
		unsigned int* histogram_r = (unsigned int*)calloc(bitDepth, sizeof(int));
		unsigned int* histogram_g = (unsigned int*)calloc(bitDepth, sizeof(int));
		unsigned int* histogram_b = (unsigned int*)calloc(bitDepth, sizeof(int));

#pragma omp for schedule(static)
		for (unsigned int i = 0; i < pic->size; i += 3)
		{
			histogram_r[pic->bitmap[i]]++;
			histogram_g[pic->bitmap[i + 1]]++;
			histogram_b[pic->bitmap[i + 2]]++;
		}

		for (unsigned int i = 0; i < bitDepth; i++)
		{
		#pragma omp atomic
			histogramR[i] += histogram_r[i];
		#pragma omp atomic
			histogramG[i] += histogram_g[i];
		#pragma omp atomic
			histogramB[i] += histogram_b[i];
		}

		free(histogram_r);
		free(histogram_g);
		free(histogram_b);
	}

	unsigned char minElValue = minElFinding(histogramR, ignoredPixelQuantity);
	unsigned char maxElValue = maxElFinding(histogramR, ignoredPixelQuantity);
	unsigned char minElValueG = minElFinding(histogramG, ignoredPixelQuantity);
	unsigned char maxElValueG = maxElFinding(histogramG, ignoredPixelQuantity);
	unsigned char minElValueB = minElFinding(histogramB, ignoredPixelQuantity);
	unsigned char maxElValueB = maxElFinding(histogramB, ignoredPixelQuantity);

	if (minElValueG < minElValue)
		minElValue = minElValueG;
	if (minElValueB < minElValue)
		minElValue = minElValueB;

	if (maxElValueG > maxElValue)
		maxElValue = maxElValueG;
	if (maxElValueB > maxElValue)
		maxElValue = maxElValueB;

	free(histogramR);
	free(histogramG);
	free(histogramB);

	if (minElValue == maxElValue)
		return 0;

	if (minElValue == 255 || maxElValue == 0)
		return 1;

	if (minElValue == 0 && maxElValue == 255)
		return 0;

	int maxMinDelta = maxElValue - minElValue;

#pragma omp parallel
	{
#pragma omp for schedule(static)
		for (unsigned int i = 0; i < pic->size; i++)
		{
			if (pic->bitmap[i] <= minElValue)
			{
				pic->bitmap[i] = 0;
				continue;
			}

			if (pic->bitmap[i] >= maxElValue)
			{
				pic->bitmap[i] = 255;
				continue;
			}

			pic->bitmap[i] = ((pic->bitmap[i] - minElValue) * 255 / maxMinDelta);
		}
	}

	return 0;
}

int main(int argc, char* argv[])
{
	if (argc == 5)
	{
		short int numOfThreads = atoi(argv[3]);
		const float ignoreRate = atof(argv[4]);

		if (numOfThreads < -1)
		{
			fprintf(stderr, "Incorrect number of threads!\n");
			return 1;
		}

		if (0.0f > ignoreRate || ignoreRate >= 0.5f)
		{
			fprintf(stderr, "Incorrect ignored value range!\n");
			return 1;
		}

		FILE* inputFile = fopen(argv[1], "rb");
		if (inputFile == NULL)
		{
			fprintf(stderr, "Input file open error!\n");
			return 1;
		}

		FILE* outputFile = fopen(argv[2], "wb");
		if (outputFile == NULL)
		{
			fprintf(stderr, "Output file open error!\n");
			fclose(inputFile);
			return 1;
		}

		struct picture pic;

		if (picSizeFinding(inputFile, &pic) == 1)
		{
			fprintf(stderr, "Invalid file format!\n");
			fclose(inputFile);
			fclose(outputFile);
			return 1;
		}

		if (5 > pic.type || pic.type > 6)
		{
			fprintf(stderr, "Invalid picture type!\n");
			fclose(inputFile);
			fclose(outputFile);
			return 1;
		}

		if (pic.width < 1 || pic.height < 1)
		{
			fprintf(stderr, "Invalid picture size!\n");
			fclose(inputFile);
			fclose(outputFile);
			return 1;
		}

		if (pic.type == 5)
			pic.size = pic.width * pic.height;

		if (pic.type == 6)
			pic.size = pic.width * pic.height * 3;

		pic.bitmap = (unsigned char*)malloc(sizeof(char) * pic.size);
		if (pic.bitmap == NULL)
		{
			fprintf(stderr, "Insufficient memory available!\n");
			free(pic.bitmap);
			fclose(inputFile);
			fclose(outputFile);
			return 1;
		}

		if (readFile(inputFile, pic.bitmap, pic.size))
		{
			fprintf(stderr, "Invalid file format!\n");
			free(pic.bitmap);
			fclose(inputFile);
			fclose(outputFile);
			return 1;
		}

		if (numOfThreads == 0)
			numOfThreads = omp_get_max_threads();

		double startRuntime, endRuntime;
		unsigned short int actualNumberOfThreads = 1;
		unsigned char contrastCorrectionSuccess = 0;

		if (numOfThreads == -1)
		{
			startRuntime = omp_get_wtime();
			if (pic.type == 5)
				contrastCorrectionSuccess = greyContrastCorrection(&pic, ignoreRate);
			else
				contrastCorrectionSuccess = anyContrastCorrection(&pic, ignoreRate);
			endRuntime = omp_get_wtime();
		}
		else
		{
			omp_set_num_threads(numOfThreads);
			startRuntime = omp_get_wtime();
			if (pic.type == 5)
				contrastCorrectionSuccess = greyParallelContrastCorrection(&pic, ignoreRate);
			else
				contrastCorrectionSuccess = anyParallelContrastCorrection(&pic, ignoreRate);
			endRuntime = omp_get_wtime();
			actualNumberOfThreads = numOfThreads;
		}

		if (contrastCorrectionSuccess)
		{
			fprintf(stderr, "Unable to correct contrast!\n");
			free(pic.bitmap);
			fclose(inputFile);
			fclose(outputFile);
			return 1;
		}

		printf("Time (%i thread(s)): %g ms\n", actualNumberOfThreads, (endRuntime - startRuntime) * 1000);

		if (writeFile(outputFile, &pic))
		{
			fprintf(stderr, "File write error!\n");
			free(pic.bitmap);
			fclose(inputFile);
			fclose(outputFile);
			return 1;
		}

		free(pic.bitmap);
		fclose(inputFile);
		fclose(outputFile);
	}
	else
	{
		fprintf(stderr, "Not enough arguments!\n");
		return 1;
	}
}
