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

public class evaluationTrainTest {

	public static void learn() throws Exception {

		String data = Globals.getTrainFile();

		if (data.isEmpty()) {
			System.err.println("evaluation: No Training File given");
			System.exit(-1);
		}

		File sourceFileTrain;
		sourceFileTrain = new File(data);
		if (!sourceFileTrain.exists()) {
			System.err.println("Train evaluation: File " + data + " not found!");
			System.exit(-1);
		}

		if (Globals.isVerbose()) {
			System.out.println("Training Source File is at: " + sourceFileTrain.getAbsolutePath());
		}

		data = Globals.getTestFile();

		if (data.isEmpty()) {
			System.err.println("evaluation: No Testing File given");
			System.exit(-1);
		}

		File sourceFileTest;
		sourceFileTest = new File(data);
		if (!sourceFileTest.exists()) {
			System.err.println("Test evaluation: File " + data + " not found!");
			System.exit(-1);
		}

		if (Globals.isVerbose()) {
			System.out.println("Testing Source File is at: " + sourceFileTest.getAbsolutePath());
		}
		
		data = Globals.getCvFile();

		File sourceFileCV;
		sourceFileCV = new File(data);
		if (!sourceFileCV.exists()) {
			System.out.println("CV evaluation: File " + data + " not found!");
		} else {
			Globals.setCvFilePresent(true);
		}

		if (Globals.isVerbose()) {
			System.out.println("CV Source File is at: " + sourceFileCV.getAbsolutePath());
		}

		Globals.setCVFILE(sourceFileCV);
		Globals.setSOURCEFILE(sourceFileTrain);

		if (!Globals.getDiscretization().equalsIgnoreCase("None") || Globals.isNormalizeNumeric() || Globals.isDoCrossValidateTuningParameter()) {

			sourceFileTrain = PreprocessData.preProcessData();

			Globals.setSOURCEFILE(sourceFileTrain);
		}

		if (!Globals.isNumInstancesKnown()) {
			Globals.setNumberInstances(SUtils.determineNumData());
		}

		Instances structure = SUtils.setStructure();

		int N = (int) Globals.getNumberInstances();
		int nc = Globals.getNumClasses();

		System.out.println("<num data points, num classes> = <" + N + ", " + nc + ">");

		double m_RMSE = 0;
		double m_Error = 0;
		double m_LogLoss = 0;

		int NTest = 0;

		double[][] instanceProbs = new double[N][nc];

		double trainTime = 0, testTime = 0;
		double trainStart = 0, testStart = 0;

		Model learner = null;

		/* ------------------------------------- */
		/* Chop-chop with the normal training   */
		/* ------------------------------------- */

		String val = Globals.getModel();

		if (val.equalsIgnoreCase("AnDE")) {

			learner = new ande();

		} 

		/* ***************************** */
		trainStart = System.currentTimeMillis();
		learner.buildClassifier();
		System.gc();
		trainTime += (System.currentTimeMillis() - trainStart);
		/* ***************************** */

		testStart = System.currentTimeMillis();

		int lineNo = 0;

		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFileTest), Globals.getBUFFER_SIZE()), Globals.getARFF_BUFFER_SIZE());
		Instance current = null;

		while ((current = reader.readInstance(structure)) != null) {

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

			NTest++;

			lineNo++;
			//System.out.println(lineNo + "," + pred);
		}

		testTime += System.currentTimeMillis() - testStart;

		System.out.print("\nTrain Test Experimentation\n");
		System.out.print("\nClassifier	   : " + Globals.getModel() + " (K = " + Globals.getLevel() + ")");
		System.out.print( "\nData File   : " + data);
		System.out.print("\nError                 : " + Utils.doubleToString(m_Error / NTest, 6, 4));
		System.out.print("\nRMSE               : " + Utils.doubleToString(Math.sqrt(m_RMSE / NTest), 6, 4));
		System.out.print("\nLogLoss           : " + Utils.doubleToString(m_LogLoss / (nc * NTest), 6, 4));
		System.out.print("\nTraining Time   : " + Utils.doubleToString(trainTime/1000, 6, 4));
		System.out.print("\nTesting Time    : " + Utils.doubleToString(testTime/1000, 6, 4));
		System.out.print("\n\n\n");


	} // endsevaluationTrainTest



}
