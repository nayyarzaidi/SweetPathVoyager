package evaluation;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.LinkedList;
import java.util.Queue;

import org.apache.commons.math3.random.MersenneTwister;

import moa.classifiers.Classifier;
import moa.classifiers.bayes.NaiveBayes;
import moa.classifiers.core.driftdetection.DDM;
import moa.classifiers.core.driftdetection.EDDM;
import moa.classifiers.drift.DriftDetectionMethodClassifier;
import moa.classifiers.functions.SGDMultiClass;
import moa.classifiers.meta.AccuracyUpdatedEnsemble;
import moa.classifiers.meta.AccuracyWeightedEnsemble;
import moa.classifiers.meta.LeveragingBag;
import moa.classifiers.meta.OzaBag;
import moa.classifiers.meta.OzaBagAdwin;
import moa.classifiers.meta.OzaBoost;
import moa.classifiers.meta.OzaBoostAdwin;
import moa.classifiers.trees.ASHoeffdingTree;
import moa.classifiers.trees.HoeffdingAdaptiveTree;
import moa.classifiers.trees.HoeffdingTree;
import moa.streams.ArffFileStream;
import utils.Globals;
import utils.SUtils;
import utils.Sampler;
import utils.filters.ida.IDADiscretizer;
import utils.filters.ida.IDADiscretizer.IDAType;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader.ArffReader;

import model.Model;
import model.ande.ande;

public class evaluationExternal {

	static DecimalFormat df = new DecimalFormat("#.####");

	public static void learn() throws Exception {

		String option = "generateAddNoiseSplit_Run";

		if (option.equalsIgnoreCase("runLibFM")) {

			/* 
			 * Create Data in LibFM format,
			 * Run LibFM
			 */

		} else if (option.equalsIgnoreCase("generateAddNoiseSplit_Run")) {

			/* 
			 * Generate Data from Drift Distribution
			 * Add noise, split it into train and test files, 
			 * and do train-test Experiment
			 */
			File sourceFile = null;
			File sourceFileTrain = null;
			File sourceFileTest = null;

			double m_RMSE = 0;
			double m_Error = 0;

			int NTest = 0;

			Model learner = null;

			double[] result = new double[Globals.getNumExp()];

			for (int exp = 0; exp < Globals.getNumExp(); exp++) {

				//sourceFile = Sampler.generateSimpleDrift(exp, 0);
				//sourceFile = Sampler.generateTANDrift(exp, 0);

				if (Globals.getDriftType().equalsIgnoreCase("simplest")) {

					//sourceFile = Sampler.generateSimpleDrift(exp, driftMagnitude);			

					System.out.println("Calling TAN Drift generator");
					sourceFile = Sampler.generateTANDrift(exp, 0);

				} else if (Globals.getDriftType().equalsIgnoreCase("simplestKDB")) {

					System.out.println("Calling KDB (K=2) Drift generator");
					sourceFile = Sampler.generateKDBDrift(exp, 0.0);		
				}

				int numNoiseColumns = Globals.getNumRandAttributes();
				sourceFile = SUtils.addNoise(numNoiseColumns, sourceFile);

				Globals.setExperimentType("preProcess");
				Globals.setPreProcessParameter("Dice");
				Globals.setDicedPercentage(50);
				Globals.setDicedStratified(false);

				Globals.setTrainFile(sourceFile.getAbsolutePath());
				Globals.setDataSetName("temp"+exp);

				evaluationPreprocess.learn();

				sourceFileTrain = new File(Globals.getTempDirectory()+"temp"+exp+"_Train.arff");
				sourceFileTest = new File(Globals.getTempDirectory()+"temp"+exp+"_Test.arff");

				String val = Globals.getModel();

				if (val.equalsIgnoreCase("AnDE")) {

					learner = new ande();

				} 

				Globals.setSOURCEFILE(sourceFileTrain);

				if (!Globals.isNumInstancesKnown()) {
					Globals.setNumberInstances(SUtils.determineNumData());
				}

				Instances structure = SUtils.setStructure();

				int N = (int) Globals.getNumberInstances();
				int nc = Globals.getNumClasses();

				learner.buildClassifier();

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

					NTest++;
				}

				result[exp] = m_Error/NTest;

			} // ends exp

			System.out.print("\nTrain Test Experimentation\n");
			System.out.print("\nClassifier	   : " + Globals.getModel() + " (K = " + Globals.getLevel() + ")");
			System.out.print( "\n SourceFile Train   : " + sourceFileTrain);
			System.out.print( "\n SourceFile Test   : " + sourceFileTest);
			System.out.print("\nError                 : " + Utils.doubleToString(m_Error / NTest, 6, 4));
			System.out.print("\nRMSE               : " + Utils.doubleToString(Math.sqrt(m_RMSE / NTest), 6, 4));
			System.out.print("\n\n\n");

		} 

	} 

}
