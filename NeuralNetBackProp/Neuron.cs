using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetBackProp
{
    public class Neuron
    {
        public const double LEARNING_RATE = 10;
        public static Random rng = new Random();
        public List<KeyValuePair<Neuron, double>> Axxons = new List<KeyValuePair<Neuron, double>>();
        public double value = 0.0;
        public Neuron()
        { }

        public Neuron(double value)
        {
            this.value = value;
        }

        public Neuron(Neuron[] connections)
        {
            foreach (Neuron n in connections)
            {
                Axxons.Add(new KeyValuePair<Neuron, double>(n, rng.NextDouble()));
            }
        }

        public async Task CalculateValue()
        {
            await Task.Run(() =>
            {
                double tempValue = 0.0;
                foreach (var kvp in Axxons)
                {
                    tempValue += kvp.Value * kvp.Key.value;
                }

                value = 1.0 / (1.0 + Math.Pow(Math.E, -tempValue));
            });
        }

        public double Part1(double expected)
        {
            return value - expected;
        }

        public double Part2()
        {
            return value * (1 - value);
        }

        public double Part3(Neuron previous)
        {
            return previous.value;
        }

        public string WeightsToString()
        {
            StringBuilder sb = new StringBuilder();
            foreach (KeyValuePair<Neuron, double> axxon in Axxons)
            {
                sb.Append(axxon.Value);
                sb.Append("\r\n");
            }
            return sb.ToString();
        }

        public void UpdateOutputAxxons(double expected)
        {
            for (int i = 0; i < Axxons.Count; i++)
            {
                double partDiff = Part1(expected) * Part2() * Part3(Axxons[i].Key);
                double newWeight = Axxons[i].Value - partDiff * LEARNING_RATE;
                Axxons[i] = new KeyValuePair<Neuron, double>(Axxons[i].Key, newWeight);
            }
        }

        //public void backPropagate(double derrivativeRate)
        //{
        //    for (int i = axxons.Count - 1; i > 0; i--)
        //    {
        //        var kvp = axxons[i];
        //        var der = (1.0 / (Math.Pow((-kvp.Value + Math.E + 1.0), 2.0)));
        //        kvp.Key.backPropagate(der);
        //        double newValue = kvp.Value - derrivativeRate * Program.learningRate;
        //        axxons[i] = new KeyValuePair<Neuron, double>(kvp.Key, newValue);
        //    }

        //}
    }
}
