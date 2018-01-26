package evaluation;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;

import model.Model;
import model.ande.ande;
import utils.Globals;
import utils.SUtils;
import utils.SanctityCheck;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

public class evaluationFlowMachines {

	public static void learn() throws Exception {

		/* ------------------------------------- */
		/* Extract Flow Information                      */
		/* ------------------------------------- */

		String flowVal = SanctityCheck.determineFlowVal();

		if (flowVal.equalsIgnoreCase("adaptiveControlParameter")) {

		} else if (flowVal.equalsIgnoreCase("sgdTuningParameter")) {

		} else if (flowVal.equalsIgnoreCase("lambda")) {

		} else {
			System.out.println("Can only create flows with {adaptiveControlParameter, sgdTuningParameter, lambda} parameters.");
			System.exit(-1);
		}

		double[] flowValues = SanctityCheck.getFlowValues();
		int numFlows = flowValues.length;

		if (flowValues == null) {
			System.out.println("Specify a proper range for flow");
			System.exit(-1);
		} else {
			System.out.println("FlowVal = " + flowVal + ", Values = " + Arrays.toString(flowValues));
		}

		/* ------------------------------------- */
		/* Ensembling starts here                        */
		/* ------------------------------------- */

		String data = Globals.getTrainFile();

		if (data.isEmpty()) {
			System.err.println("evaluation: No Training File given");
			System.exit(-1);
		}

		File sourceFile = null;
		int N = 0;
		Instances structure = null;

		/* Read training data from the file */

		sourceFile = new File(data);
		if (!sourceFile.exists()) {
			System.err.println("evaluation: File " + data + " not found!");
			System.exit(-1);
		}

		if (Globals.isVerbose()) {
			System.out.println("Initial Source File is at: " + sourceFile.getAbsolutePath());
		}

		Globals.setSOURCEFILE(sourceFile);

		if (!Globals.isNumInstancesKnown()) {
			Globals.setNumberInstances(SUtils.determineNumData());
		}

		structure = SUtils.setStructure();

		N = (int) Globals.getNumberInstances();
		int nc = Globals.getNumClasses();
		System.out.println("<num data points, num classes> = <" + N + ", " + nc + ">");

		ArrayList<ArrayList<ArrayList<Double>>> rmseExpFlowResults = new ArrayList<ArrayList<ArrayList<Double>>>(); 

		Model[] learners = null;

		int numDataEvaluated = 0;

		/* ------------------------------------- */
		/* Chop-chop with the normal training   */
		/* ------------------------------------- */

		for (int exp = 0; exp < Globals.getNumExp(); exp++) {

			ArrayList<ArrayList<Double>> rmseExpResults = new ArrayList<ArrayList<Double>>();

			sourceFile = SUtils.randomizeTrainingFile();

			Globals.setSOURCEFILE(sourceFile);

			if (Globals.isVerbose()) {
				System.out.println("Experiment No. " + exp);
			}

			String val = Globals.getModel();

			if (val.equalsIgnoreCase("AnDE")) {

				learners = new Model[numFlows];
				for (int f = 0; f < numFlows; f++) {
					learners[f] = new ande();
					learners[f].buildClassifier();
				}

			} else if (val.equalsIgnoreCase("ALR")) {

			} else if (val.equalsIgnoreCase("KDB")) {

			} else if (val.equalsIgnoreCase("FM")) {

			} else if (val.equalsIgnoreCase("ANN")) {

			}

			ArffReader reader = new ArffReader(new BufferedReader(new FileReader(Globals.getSOURCEFILE()), Globals.getBUFFER_SIZE()), Globals.getARFF_BUFFER_SIZE());
			Instance row = null;

			int lineNo = 0;
			N = 0;
			numDataEvaluated = 0;

			while ((row = reader.readInstance(structure)) != null) {

				if (lineNo % Globals.getPrequentialOutputResolution() == 0) {

					ArrayList<Double> rmseResults = new ArrayList<Double>();

					for (int f = 0; f < numFlows; f++) {

						double[] probs = learners[f].distributionForInstance(row);

						double[] results = SUtils.getResults(probs, (int) row.classValue(), nc);

						rmseResults.add(results[0]);
					}

					rmseExpResults.add(rmseResults);
					numDataEvaluated++;
				}

				for (int f = 0; f < numFlows; f++) {

					if (flowVal.equalsIgnoreCase("adaptiveControlParameter")) {
						Globals.setAdaptiveControlParameter(flowValues[f]);
					} else if (flowVal.equalsIgnoreCase("sgdTuningParameter")) {
						Globals.setSgdTuningParameter(flowValues[f]);
					} else if (flowVal.equalsIgnoreCase("lambda")) {
						Globals.setLambda(flowValues[f]);
					} 

					learners[f].update(row);
				}

				lineNo++;
				N++;
				Globals.setNumberInstances(N);
			}

			System.out.println("\nExp: " + exp + ". Read: " + lineNo + " data points, out of which learner was evaluated on: " + numDataEvaluated);

			rmseExpFlowResults.add(rmseExpResults);

		} // ends exp

		double[][] averageResults = new double[numDataEvaluated][numFlows];

		for (int exp = 0; exp < Globals.getNumExp(); exp++) {
			for (int i = 0; i < numDataEvaluated; i++) {
				for (int f = 0; f < numFlows; f++) {
					averageResults[i][f] += ((double)1/Globals.getNumExp() * rmseExpFlowResults.get(exp).get(i).get(f));
				}
			}
		}

		if (Globals.isDoMovingAverage()) {

		} else {

			ArrayList<ArrayList<Double>> buffAverageResults = new ArrayList<ArrayList<Double>>();

			int bufferSize = Globals.getPrequentialBufferOutputResolution();

			for (int i = 0; i < numDataEvaluated; i++) {
				ArrayList<Double> flowAverageResults = new ArrayList<Double>(); 
				for (int f = 0; f < numFlows; f++) {
					if (i < bufferSize) {
						//flowAverageResults.add(averageResults[i][f]);
					} else {
						double tempSum = 0;
						for (int j = i; j < i + bufferSize; j++) {
							if (j >= numDataEvaluated) {
								break;
							}
							tempSum += averageResults[j][f];
						}
						flowAverageResults.add((double)1/bufferSize * tempSum);
						i += bufferSize;
					}
				}
				buffAverageResults.add(flowAverageResults);
			}

			String identifier = Globals.getModel() + "_" + Globals.getLevel();

			String outputFile = Globals.getOuputResultsDirectory() + identifier + "_LC.m";

			File file = new File(outputFile);

			BufferedWriter output = null;
			output = new BufferedWriter(new FileWriter(file));

			for (int f = 0; f < numFlows; f++) {
				output.write("fx_" + identifier + "_f_" + flowValues[f] + " = [");
				for (int i = 0; i < buffAverageResults.size() - 1; i++) {
					output.write(buffAverageResults.get(i).get(f) + ", ");
				}
				output.write("];\n");
			}

			output.close();


		}


	}

}
