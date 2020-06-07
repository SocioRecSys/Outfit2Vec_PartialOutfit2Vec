import argparse
from WholeOutfitsPrediction import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataFile', dest='path_data',  type=str, default='/data/OutfitsRecommendation/Data/categoriesOrdered')
    parser.add_argument('--k', dest='k', default=5, type=int)
    parser.add_argument('--window', dest='window', default=5, type=int)
    parser.add_argument('--learning_rate', dest='learning_rate', default=0.025, type=float)
    parser.add_argument('--testSize', dest='testSize', default=0.3, type=float)
    parser.add_argument('--dm', dest='dm', default=1, type=int)
    parser.add_argument('--vector_size', dest='vector_size', default=300, type=int)
    parser.add_argument('--resultsFilePath', dest='resultsFilePath', default='/results/writeResults.txt', type=str)
    parser.add_argument('--isStructured', dest='isStructured', default=True, type=bool)
    parser.add_argument('--methodNumber', dest='methodNumber', default=3, type=int)
    parser.add_argument('--epochs', dest='epochs', default=30, type=int)
    args = parser.parse_args()

    obj = WholeOutfitsPredictor(args.k, args.window, args.learning_rate, args.dataFile, args.testSize, args.dm,
                         args.vector_size, args.resultsFilePath, args.isStructured, args.methodNumber, args.epochs)
    obj.train()