using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetBackProp
{
    public static class Library
    {
        public static double IndexToFraction(this char[] total,int index)
        {
            int plusOne = index + 1;
            double result = (double)index / (double)total.Length;
            return result;
        }

        public static double[] BuildExpectation(this char[] vocab,int index)
        {
            List<double> result = new List<double>();
            for (int i = 0; i < vocab.Length; i++)
            {
                if (i == index)
                {
                    result.Add(1.0);
                }
                else
                {
                    result.Add(0.0);
                }
            }
            return result.ToArray();
        }

        public static string[] SplitChunks(this string str, int chunkSize)
        {
            return Enumerable.Range(0, str.Length / chunkSize)
                .Select(i => str.Substring(i * chunkSize, chunkSize)).ToArray();
        }
    }
}
