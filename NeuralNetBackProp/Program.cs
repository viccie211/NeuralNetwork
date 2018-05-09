using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetBackProp
{
    class Program
    {   
        static void Main(string[] args)
        {
            string input = System.IO.File.ReadAllText("input.txt");
            List<char>vocabList = input.ToCharArray().Distinct().ToList();//Getting all the distinct characters from the input.
            vocabList.Insert(0, '^');//Making sure that the char with index 0 is not a meaningful character.
            char[] vocab = vocabList.ToArray();

            Net net = new Net(vocab);
            net.InputA.value = 0.5;//Setting the base values
            net.InputB.value = 0.5;

            string[] trainingData = input.SplitChunks(10);

            Random random = new Random();

            for (int i = 0; i < 100000000; i++)
            {
                string training = trainingData[random.Next(trainingData.Length - 1)];//Select a random piece of training data for this iteration
                double error = 0.0;

                for (int j = 0; j < training.Length; j++)
                {
                    net.InputA.value = vocab.IndexToFraction(Array.IndexOf(vocab, training[j]));//We set net.InputA to the current char we are looking at.

                    if (j != 0)//We set net.InputB to be the result of the last forward propagation of the net unless we start the iteration.
                    {
                        char result = net.GetResult(vocab);
                        net.InputB.value = vocab.IndexToFraction(Array.IndexOf(vocab, result));
                    }
                    else
                    {
                        net.InputB.value = vocab.IndexToFraction(Array.IndexOf(vocab, '^'));
                    }

                    foreach (Neuron[] layer in net.HiddenLayersAndOutputLayer)//For every layer we calulate the values of the neurons async since neurons on the same layer can't affect eachother.
                    {
                        var tasks = new List<Task>();
                        foreach (Neuron n in layer)
                        {
                            tasks.Add(Task.Run(() => n.CalculateValue()));
                        }
                        Task.WaitAll(tasks.ToArray());
                    }

                    if (j < training.Length - 2)//if j>training.Length - 2, j+1 will give an IndexOutOfBoundsException so we can only learn when j<training.Length-2
                    {
                        double[] expected = vocab.BuildExpectation(Array.IndexOf(vocab,training[j + 1]));
                        double[] errors = net.GetError(expected);

                        for(int k =0;k<errors.Length;k++)
                        {
                            //Console.WriteLine("Value:"+Output[k].value+" Expected:"+expected[k]+" Error:"+errors[k]);
                            Console.WriteLine(net.OutputLayer[k].WeightsToString());
                            net.OutputLayer[k].UpdateOutputAxxons(expected[k]);
                            Console.WriteLine("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
                            Console.WriteLine(net.OutputLayer[k].WeightsToString());
                        }
                        Console.Read();
                        //Console.Write(GetResult(Output, vocab));
                    }
                }
            }
            Console.Read();
        }
    }
}
