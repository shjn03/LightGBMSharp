using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;

namespace LightGBMSharp
{


    class Program
    {
        static void Main(string[] args)

        {
            string model_path = "../../model/regression_model.txt";
            string data_path = "../../../data/regression/regression.test";
            string out_filepath = "prediction_result_cs.csv";
            string parameters = "";
            int ret;
            var sw = new System.Diagnostics.Stopwatch();
            
            sw.Start();
            ret = BoosterMethods.LGBM_BoosterCreateFromModelfile(model_path, out int num_models, out BoosterHandle booster);
            sw.Stop();
            if (ret != 0)
            {
                var err = Marshal.PtrToStringAnsi(NativeMethods.LGBM_GetLastError());
                throw new Exception(err);
            }
            else {
                Console.WriteLine($"Successfully load model from {model_path}");
                Console.WriteLine($"Elapsed {sw.ElapsedMilliseconds}ms");
            }
            DatasetHandle reference = null;
            reference = reference ?? DatasetHandle.Zero;
            ret = DatasetMethods.LGBM_DatasetCreateFromFile(data_path, parameters, reference, out DatasetHandle x_test);
            if (ret != 0)
            {
                var err = Marshal.PtrToStringAnsi(NativeMethods.LGBM_GetLastError());
                throw new Exception(err);
            }
            else
            {
                Console.WriteLine($"Successfully load data from {data_path}");
                //Console.WriteLine($"Elapsed {sw.ElapsedMilliseconds}ms");
            }

            DatasetMethods.LGBM_DatasetGetNumData(x_test, out int num_rows);
            DatasetMethods.LGBM_DatasetGetNumFeature(x_test, out int num_cols);
            BoosterMethods.LGBM_BoosterCalcNumPredict(booster, num_rows, LGBMPredictType.PredictNormal, num_models, out int num_predicts);
            var result = new double[num_predicts];

            sw.Restart();
            ret = BoosterMethods.LGBM_BoosterPredictForFile(booster, data_path, false, LGBMPredictType.PredictNormal, num_models, parameters, out_filepath);
            sw.Stop();
            if (ret != 0)
            {
                var err = Marshal.PtrToStringAnsi(NativeMethods.LGBM_GetLastError());
                throw new Exception(err);
            }
            else
            {
                Console.WriteLine($"Successfully predict(from file) result is saved to {out_filepath}");
                Console.WriteLine($"Elapsed {sw.ElapsedMilliseconds}ms");
            }


            sw.Restart();
            ret = BoosterMethods.LGBM_BoosterPredictForMat(booster, x_test.DangerousGetHandle(), LGBMDataType.Float32, num_rows, num_cols,
                                                           0, LGBMPredictType.PredictNormal, num_models, parameters,
                                                           out int outputlen, result) ;
            sw.Stop();
            if (ret != 0)
            {
                var err = Marshal.PtrToStringAnsi(NativeMethods.LGBM_GetLastError());
                throw new Exception(err);
            }else
            { 
                Console.WriteLine($"Successfully predict(from mat)");
                Console.WriteLine($"Elapsed {sw.ElapsedMilliseconds}ms");
            }


            Console.ReadLine();
            booster.Close();
            x_test.Close();
            


        }
    }
}
