using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetBackProp
{
    public class Net
    {
        private const int hiddenLayersAmount = 8;
        private const int neuronsPerLayer = 20;

        public Neuron[][] HiddenLayersAndOutputLayer;
        public Neuron[] OutputLayer;
        public Neuron InputA;
        public Neuron InputB;
        
        public Net(char[] vocab)
        {
            InputA = new Neuron();
            InputB = new Neuron();
            List<Neuron[]> allLayersMinusInput = new List<Neuron[]>();

            for (int i = 0; i < hiddenLayersAmount; i++)
            {
                List<Neuron> layer = new List<Neuron>();
                for (int j = 0; j < neuronsPerLayer; j++)
                {
                    if (i == 0)
                    {
                        layer.Add(new Neuron(new Neuron[] { InputA, InputB }));
                    }
                    else
                    {
                        layer.Add(new Neuron(allLayersMinusInput[i - 1]));
                    }
                }
                allLayersMinusInput.Add(layer.ToArray());
            }
            List<Neuron> outputList = new List<Neuron>();
            foreach (char c in vocab)
            {
                var n = new Neuron(allLayersMinusInput.Last());
                outputList.Add(n);
            }
            OutputLayer = outputList.ToArray();
            allLayersMinusInput.Add(OutputLayer);
            HiddenLayersAndOutputLayer = allLayersMinusInput.ToArray();
        }

        public char GetResult(char[] vocab)
        {
            double highestValue = double.MinValue;
            int highestIndex = -1;
            for (int i = 0; i < OutputLayer.Length && i < vocab.Length; i++)
            {
                if (OutputLayer[i].value > highestValue)
                {
                    highestValue = OutputLayer[i].value;
                    highestIndex = i;
                }
            }
            return vocab[highestIndex];
        }

        public double[] GetError(double[] expected)
        {
            double[] result = new double[expected.Length];
            for (int i = 0; i < OutputLayer.Length && i < expected.Length && i < result.Length; i++)
            {
                result[i] = 0.5 * (Math.Pow(OutputLayer[i].value - expected[i], 2));
            }
            return result;
        }
    }
}
