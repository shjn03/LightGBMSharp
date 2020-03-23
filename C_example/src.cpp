#include <LightGBM/c_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
//simple predict exmaple

int main() {

	int ret;
	//DatasetHandle dataset_train;
	DatasetHandle x_test;
	BoosterHandle Booster;
	int num_models;
	//const char *train_path = "../data/regression/regression.train";
	const char *test_path = "../data/regression/regression.test";
	const char *dataset_parameters = "";
	const char* model_path = "model/regression_model.txt";
	const char* booster_parameters = "";
	const char* out_filename = "pred_result.csv";
	int n_rows;
	int n_cols;
	long long num_predictions;
	double time;
	// ŽžŠÔŒv‘ª—p
	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);
	LARGE_INTEGER start, end;

	//Load dataset from csv file 
	ret = LGBM_DatasetCreateFromFile(test_path, dataset_parameters, nullptr, &x_test);
	if (0 == ret) {
		printf("Successfully load test dataset.\n");

	}
	else {
		printf("failed to load test dataset.\n");
		printf("%s\n", LGBM_GetLastError());
	}

	//get data discription (for use prediction from mat)
	ret = LGBM_DatasetGetNumData(x_test, &n_rows);
	ret = LGBM_DatasetGetNumFeature(x_test, &n_cols);
	double* pred = (double*)malloc(n_rows * sizeof(double));

	//load pretrained model
	ret = LGBM_BoosterCreateFromModelfile(model_path, &num_models, &Booster);
	if (0 == ret) {
		printf("Successfully load model.\n");
		}
	else {
		printf("failed to load model.\n");
		printf("%s\n", LGBM_GetLastError());
	}

	//predict from csv file
	QueryPerformanceCounter(&start);
	ret = LGBM_BoosterPredictForFile(Booster, test_path, 0, 0, num_models, booster_parameters, out_filename);
	QueryPerformanceCounter(&end);
	if (0 == ret) {
		printf("Successfully predict from file.\n");
		time = static_cast<double>(end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
		printf("elapsed time %lf[ms]\n", time);

	}
	else {
		printf("failed to predict.\n");
		printf("%s\n", LGBM_GetLastError());
	}

	if (n_rows == 1) {
		//predict from Dataset(Single row mode)
		QueryPerformanceCounter(&start);
		ret = LGBM_BoosterPredictForMatSingleRow(Booster, x_test, 0, n_cols, 0, 0, num_models, booster_parameters, &num_predictions, pred);
		QueryPerformanceCounter(&end);
		if (0 == ret) {
			printf("Successfully predict from mat(single row).\n");
			time = static_cast<double>(end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
			printf("elapsed time %lf[ms]\n", time);
		}
		else {
			printf("failed to predict.\n");
			printf("%s\n", LGBM_GetLastError());
		}
	}else{

		//predict from Dataset
		QueryPerformanceCounter(&start);
		ret = LGBM_BoosterPredictForMat(Booster, x_test, 0, n_rows, n_cols, 0, 0, num_models, booster_parameters, &num_predictions, pred);
		QueryPerformanceCounter(&end);
		if (0 == ret) {
			printf("Successfully predict from mat.\n");
			time = static_cast<double>(end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
			printf("elapsed time %lf[ms]\n", time);
		}
		else {
			printf("failed to predict.\n");
			printf("%s\n", LGBM_GetLastError());
		}
	}



	LGBM_DatasetFree(x_test);
	LGBM_BoosterFree(Booster);
	free(pred);
	getchar();
	return 0;

}