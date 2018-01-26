package evaluation;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.LinkedList;
import java.util.Queue;

import org.apache.commons.math3.random.MersenneTwister;

import utils.Globals;
import utils.SUtils;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader.ArffReader;
import model.Model;
import model.ande.ande;

public class evaluationCV {

	public static void learn() throws Exception {

		String data = Globals.getTrainFile();

		if (data.isEmpty()) {
			System.err.println("evaluation: No Training File given");
			System.exit(-1);
		}

		File sourceFile;
		sourceFile = new File(data);
		if (!sourceFile.exists()) {
			System.err.println("evaluation: File " + data + " not found!");
			System.exit(-1);
		}

		if (Globals.isVerbose()) {
			System.out.println("Initial Source File is at: " + sourceFile.getAbsolutePath());
		}


		Globals.setSOURCEFILE(sourceFile);

		if (!Globals.getDiscretization().equalsIgnoreCase("None") || Globals.isNormalizeNumeric() || Globals.isDoCrossValidateTuningParameter()) {

			sourceFile = PreprocessData.preProcessData();

			Globals.setSOURCEFILE(sourceFile);
		}

		if (!Globals.isNumInstancesKnown()) {
			Globals.setNumberInstances(SUtils.determineNumData());
		}

		//Instances structure = SUtils.getStructure();
		Instances structure = SUtils.setStructure();

		int N = (int) Globals.getNumberInstances();
		int nc = Globals.getNumClasses();

		System.out.println("<num data points, num classes> = <" + N + ", " + nc + ">");

		double m_RMSE = 0;
		double m_Error = 0;
		double m_LogLoss = 0;

		int NTest = 0;
		long seed = 3071980;

		double[][] instanceProbs = new double[N][nc];

		double trainTime = 0, testTime = 0;
		double trainStart = 0, testStart = 0;

		Model learner = null;

		/* ------------------------------------- */
		/* Chop-chop with the normal training   */
		/* ------------------------------------- */

		for (int exp = 0; exp < Globals.getNumExp(); exp++) {

			if (Globals.isVerbose()) {
				System.out.println("Experiment No. " + exp);
			}

			seed++;

			BitSet[] indexes = new BitSet[Globals.getNumFolds()];
			for (int i = 0; i < Globals.getNumFolds(); i++) {
				indexes[i] = new BitSet();
			}

			SUtils.getIndexes(indexes);

			for (int fold = 0; fold < Globals.getNumFolds(); fold++) {

				if (Globals.isVerbose()) {
					System.out.println("Fold No. " + fold);
				}

				BitSet trainIndexes = SUtils.combineIndexes(indexes, fold);

				File trainFile = SUtils.createTrainTmpFile(structure, trainIndexes);

				System.out.println("Train file generated");

				if (Globals.isVerbose()) {
					System.out.println("Training fold " + fold +": trainFile is '" + trainFile.getAbsolutePath() + "'");
				}

				String val = Globals.getModel();

				if (val.equalsIgnoreCase("AnDE")) {

					learner = new ande();

				} 

				/* ***************************** */
				trainStart = System.currentTimeMillis();
				Globals.setSOURCEFILE(trainFile);
				learner.buildClassifier();
				Globals.setSOURCEFILE(sourceFile);
				trainTime += (System.currentTimeMillis() - trainStart);
				/* ***************************** */

				if (Globals.isVerbose()) {
					System.out.println("Testing fold 0 started");
				}

				testStart = System.currentTimeMillis();

				int thisNTest = 0;
				int lineNo = 0;

				ArffReader reader = new ArffReader(new BufferedReader(new FileReader(Globals.getSOURCEFILE()), Globals.getBUFFER_SIZE()), Globals.getARFF_BUFFER_SIZE());
				Instance current = null;

				while ((current = reader.readInstance(structure)) != null) {
					if (!trainIndexes.get(lineNo)) {
						double[] probs = new double[nc];
						int x_C = (int) current.classValue();

						probs = learner.distributionForInstance(current);	

						// ------------------------------------
						// Update Error and RMSE
						// ------------------------------------
						int pred = -1;
						double bestProb = Double.MIN_VALUE;
						for (int y = 0; y < nc; y++) {
							if (!Double.isNaN(probs[y])) {
								if (probs[y] > bestProb) {
									pred = y;
									bestProb = probs[y];
								}
								m_RMSE += (1 / (double) nc * Math.pow((probs[y] - ((y == x_C) ? 1 : 0)), 2));
							} else {
								System.err.println("probs[ " + y + "] is NaN! oh no!");
							}
						}

						if (pred != x_C) {
							m_Error += 1;
						}

						m_LogLoss += Math.log(probs[x_C]);

						thisNTest++;
						NTest++;

						instanceProbs[lineNo][pred]++;
					}
					lineNo++;
				}

				testTime += System.currentTimeMillis() - testStart;

				if (Globals.isVerbose()) {
					System.out.println("Testing fold " + fold + " finished - 0-1=" + (m_Error / NTest) + "\trmse=" + Math.sqrt(m_RMSE / NTest) + "\tlogloss=" + m_LogLoss / (nc * NTest));
				}


			} // ends fold

		} // ends exp

		double m_Bias = 0;
		double m_Sigma = 0;
		double m_Variance = 0;

		int lineNo = 0;
		Instance current = null;
		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(Globals.getSOURCEFILE()), Globals.getBUFFER_SIZE()), Globals.getARFF_BUFFER_SIZE());
		while ((current = reader.readInstance(structure)) != null) {
			double[] predProbs = instanceProbs[lineNo];

			double pActual, pPred;
			double bsum = 0, vsum = 0, ssum = 0;
			for (int j = 0; j < nc; j++) {
				pActual = (current.classValue() == j) ? 1 : 0;
				pPred = predProbs[j] / Globals.getNumExp();
				bsum += (pActual - pPred) * (pActual - pPred) - pPred * (1 - pPred) / (Globals.getNumExp() - 1);
				vsum += (pPred * pPred);
				ssum += pActual * pActual;
			}
			m_Bias += bsum;
			m_Variance += (1 - vsum);
			m_Sigma += (1 - ssum);

			lineNo++;
		}

		m_Bias = m_Bias / (2 * lineNo);
		m_Variance = (m_Error / NTest) - m_Bias;

		System.out.print("\nBias-Variance Decomposition\n");
		System.out.print("\nClassifier	   : " + Globals.getModel() + " (K = " + Globals.getLevel() + ")");
		System.out.print( "\nData File   : " + data);
		System.out.print("\nError                 : " + Utils.doubleToString(m_Error / NTest, 6, 4));
		System.out.print("\nBias^2              : " + Utils.doubleToString(m_Bias, 6, 4));
		System.out.print("\nVariance           : " + Utils.doubleToString(m_Variance, 6, 4));
		System.out.print("\nRMSE               : " + Utils.doubleToString(Math.sqrt(m_RMSE / NTest), 6, 4));
		System.out.print("\nLogLoss           : " + Utils.doubleToString(m_LogLoss / (nc * NTest), 6, 4));
		System.out.print("\nTraining Time   : " + Utils.doubleToString(trainTime/1000, 6, 4));
		System.out.print("\nTesting Time    : " + Utils.doubleToString(testTime/1000, 6, 4));
		System.out.print("\n\n\n");


	} // endsevaluationCV



}
