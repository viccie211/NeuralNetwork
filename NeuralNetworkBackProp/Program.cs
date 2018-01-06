using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkBackProp
{
    class Program
    {
        private const int TRANING_CHUNK_SIZE = 100;
        static void Main(string[] args)
        {
            string input = readInputText("input");
            string[] trainingData = Split(input,TRANING_CHUNK_SIZE);
            List<char> vocab= input.Distinct().ToList();
            vocab.Insert(0, 'Ϫ');//Add a "begin of training" character

            Net neuralNet = new Net(vocab);

            Random random = new Random();
            for(int i=0;i<10000000;i++)
            {
                neuralNet.train(trainingData[random.Next(trainingData.Length)]);
            }
        }

        private static string readInputText(string location)
        {
            return System.IO.File.ReadAllText(location);
        }

        private static string[] Split(string str, int chunkSize)
        {
            return Enumerable.Range(0, str.Length / chunkSize)
                .Select(i => str.Substring(i * chunkSize, chunkSize)).ToArray();
        }
    }

    class Net
    {
        const int HIDDEN_LAYERS_COUNT = 5;
        const int NEURONS_PER_LAYER_COUNT = 20;

        public List<Neuron[]> AllLayers { get; set; }

        public List<char> vocab { get; set; }

        public Net(List<char> vocab)
        {
            this.vocab = vocab;

            Neuron inputA = new Neuron();
            Neuron inputB = new Neuron();
            Neuron[] inputLayer = new Neuron[] { inputA, inputB };
            AllLayers = new List<Neuron[]>();
            AllLayers.Add(inputLayer);

            for(int i=0;i<HIDDEN_LAYERS_COUNT;i++)//Create the hidden layers of neurons
            {
                List<Neuron> layer = new List<Neuron>();

                for (int j=0;j<NEURONS_PER_LAYER_COUNT;j++)
                {
                    layer.Add(new Neuron(AllLayers[i]));//Since we've already added the input as 0 the layer coming before will have the index equal to i.
                }

                AllLayers.Add(layer.ToArray());
            }

            List<Neuron> outputLayer = new List<Neuron>();

            foreach (char c in vocab)
            {
                outputLayer.Add(new Neuron(AllLayers.Last()));
            }

            AllLayers.Add(outputLayer.ToArray());
        }

        public void train(string training)
        {
            for (int a = 0; a < training.Length; a++)
            {
                AllLayers[0][0].Value = indexOfAsFraction(training[a]);
                AllLayers[0][1].Value = indexOfAsFraction('Ϫ');//this needs to become the output of the last or this character when it's the first

                for (int i=1;i<AllLayers.Count;i++)//start at 1 because we need to skip the input layer
                {
                    foreach(Neuron neuron in AllLayers[i])
                    {
                        neuron.CalculateValue();
                    }
                }

                if(a!=training.Length-1)
                {
                    double[] expected = buildExpectations(training[a + 1]);

                    double error = calculateError(expected);
                    double loss = Math.Pow(error, 2);
                    double derLoss = 2 * error;
                }
                
            }
        }

        private double[] buildExpectations(char c)
        {
            List<double> result = new List<double>();

            foreach (char k in vocab)
            {
                if(k==c)
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

        private double calculateError(double[] expected)
        {
            double error = 0.0;

            for(int i=0;i<expected.Length;i++)
            {
                error += Math.Abs(expected[i] - AllLayers.Last()[i].Value);
            }

            return error;
        }

        private void backProp(double derLoss)
        {
            foreach(Neuron n in AllLayers.Last())
            {
                n.backProp(derLoss);
            }
        }

        private double indexOfAsFraction(char c)
        {
            return (double)vocab.IndexOf(c) + 1.0 / (double)vocab.Count();
        }
    }

    class Neuron
    {
        public double Value { get; set; }
        public List<KeyValuePair<Neuron, double>> Axxons { get; set; }

        public Neuron()
        {
            this.Axxons = new List<KeyValuePair<Neuron, double>>();
        }

        public Neuron(Neuron[] lastLayer)
        {
            Random random = new Random();
            this.Axxons = new List<KeyValuePair<Neuron, double>>();

            foreach (Neuron neuron in lastLayer)
            {
                double weight = random.NextDouble();
                int negOrPos = random.Next(0, 2);
                if(negOrPos==0)
                {
                    weight = -weight;
                }
                this.Axxons.Add(new KeyValuePair<Neuron, double>(neuron,weight));
            }
        }

        public void backProp(double derLoss)
        {
            //loop backwards through axxons update doubles to old - derLoss*learningRate
            //call backprop for all neurons in axxons with e^value/(e^value+1)^2
        }

        public void CalculateValue()
        {
            double tempValue = 0.0;

            foreach (KeyValuePair<Neuron,double>axxon in Axxons)
            {
                tempValue += axxon.Key.Value * axxon.Value;
            }

            Value = 1 / (1 + Math.Pow(Math.E, -tempValue));
        }
    }
}
