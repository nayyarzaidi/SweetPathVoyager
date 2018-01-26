package evaluation;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.LinkedList;
import java.util.Queue;

import utils.Globals;
import utils.SUtils;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader.ArffReader;
import model.Model;
import model.ande.ande;

public class evaluationPrequential {

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

		if (Globals.isDoCrossValidateTuningParameter()) {
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

		Model learner = null;
		String val = Globals.getModel();

		/* ------------------------------------- */
		/* Chop-chop with the normal training   */
		/* ------------------------------------- */

		ArrayList<ArrayList<Double>> rmseResults = new ArrayList<ArrayList<Double>>(); 
		int numDataEvaluated = 0;

		long seed = 3071980;

		learner = null;

		for (int exp = 0; exp < Globals.getNumExp(); exp++) {

			sourceFile = SUtils.randomizeTrainingFile();

			Globals.setSOURCEFILE(sourceFile);

			ArrayList<Double> rmseExpResults = new ArrayList<Double>();

			if (Globals.isVerbose()) {
				System.out.println("Experiment No. " + exp);
			}

			seed++;

			val = Globals.getModel();

			if (val.equalsIgnoreCase("AnDE")) {

				learner = new ande();
				learner.buildClassifier();

			} 
			
			ArffReader reader = new ArffReader(new BufferedReader(new FileReader(Globals.getSOURCEFILE()), Globals.getBUFFER_SIZE()), Globals.getARFF_BUFFER_SIZE());
			Instance row = null;

			int lineNo = 0;
			numDataEvaluated = 0;
			N = 0;
			while ((row = reader.readInstance(structure)) != null) {

				if (lineNo % Globals.getPrequentialOutputResolution() == 0) {
					double[] probs = learner.distributionForInstance(row);

					double[] results = SUtils.getResults(probs, (int) row.classValue(), nc);

					rmseExpResults.add(results[0]);
					numDataEvaluated++;
				}

				learner.update(row);

				lineNo++;
				N++;

				Globals.setNumberInstances(N);
			}

			System.out.println("\nExp: " + exp + ". Read: " + lineNo + " data points, out of which learner was evaluated on: " + numDataEvaluated);

			rmseResults.add(rmseExpResults);

		} // ends exp

		double[] averageResults = new double[numDataEvaluated];
		for (int exp = 0; exp < Globals.getNumExp(); exp++) {
			for (int i = 0; i < numDataEvaluated; i++) {
				averageResults[i] += ((double)1/Globals.getNumExp() * rmseResults.get(exp).get(i));
			}
		}

		ArrayList<Double> buffAverageResults = new ArrayList<Double>();

		int bufferSize = Globals.getPrequentialBufferOutputResolution();

		if (Globals.isDoMovingAverage()) {

			for (int i = 0; i < numDataEvaluated ; i++) {
				if (i < bufferSize) {
					//buffAverageResults.add(averageResults[i]);
				} else {
					double tempSum = 0;
					for (int j = i; j < i + bufferSize; j++) {
						if (j >= numDataEvaluated) {
							break;
						}
						tempSum += averageResults[j];
					}
					buffAverageResults.add((double)1/bufferSize * tempSum);
					i++;
				}
			}

		} else {

			for (int i = 0; i < numDataEvaluated; i++) {
				if (i < bufferSize) {
					//buffAverageResults.add(averageResults[i]);
				} else {
					double tempSum = 0;
					for (int j = i; j < i + bufferSize; j++) {
						if (j >= numDataEvaluated) {
							break;
						}
						tempSum += averageResults[j];
					}
					buffAverageResults.add((double)1/bufferSize * tempSum);
					i += bufferSize;
				}
			}

		}

		String identifier = Globals.getModel() + "_" + Globals.getLevel();

		if (Globals.getModel().equalsIgnoreCase("ALR")) {
			if (Globals.isDoWanbiac()) {
				identifier += "w";
			}

			if (Globals.isDoDiscriminative()) {
				identifier += "d";	
			}
		}

		String outputFile = Globals.getOuputResultsDirectory() + identifier + "_LC.m";

		File file = new File(outputFile);

		BufferedWriter output = null;
		output = new BufferedWriter(new FileWriter(file));

		output.write("fx_" + identifier + " = [");
		for (int i = 0; i < buffAverageResults.size() - 1; i++) {
			output.write(buffAverageResults.get(i) + ", ");
		}
		output.write("];");
		output.close();


		//		File file = new File("/Users/nayyar/Desktop/example.m");
		//
		//		BufferedWriter output = null;
		//		output = new BufferedWriter(new FileWriter(file));
		//
		//		output.write("fx = [");
		//		for (int i = 0; i < numDataEvaluated; i++) {
		//			output.write(averageResults[i] + ", ");
		//		}
		//		output.write("];");
		//		output.close();


	} // endsevaluationCV	

}
