package evaluation;

import java.awt.Color;
import java.awt.geom.Ellipse2D;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.Writer;
import java.util.Arrays;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYErrorRenderer;
import org.jfree.data.xy.XYIntervalSeries;
import org.jfree.data.xy.XYIntervalSeriesCollection;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import utils.DriftGenerator;
import utils.ChartTools;

import model.Model;
import model.ande.ande;
import utils.Globals;
import utils.Plotting;
import utils.SUtils;
import utils.Sampler;
import utils.SanctityCheck;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

public class evaluationDrift {

	public static void learn() throws Exception {

		/* ------------------------------------- */
		/* Extract Flow Information                      */
		/* ------------------------------------- */

		String flowVal = SanctityCheck.determineFlowVal();

		if (flowVal.equalsIgnoreCase("level")) {
			System.out.println("FlowVal used is: " + flowVal);

		}  else {
			System.out.println("Can only create flows with {adaptiveControlParameter} parameters.");
			System.exit(-1);
		}

		double[] flowValues = SanctityCheck.getFlowValues();

		if (flowValues == null) {
			System.out.println("Specify a proper range for flow");
			System.exit(-1);
		} else {
			System.out.println("FlowVal = " + flowVal + ", Values = " + Arrays.toString(flowValues) + "\n");
		}

		int numFlows = flowValues.length;

		/* ----------------------------------- */
		/* Initialize Parameters */
		/* ----------------------------------- */

		//double[] decayWindow_Rate = {0.1, 0.01, 0.001, 0.0001, 0.00001};
		//Globals.setAdaptiveControl("decay");
		//String dw_name = "Decay=";

		//double[] decayWindow_Rate = {-1,1, 5, 10, 20};
		
		//double[] decayWindow_Rate = {-1, 10, 20, 50, 100, 200, 500};
		
		//double[] decayWindow_Rate = {10, 20, 50, 100, 200};
		//double[] decayWindow_Rate = {-1, 10, 50, 100, 500};
		
		//double[] decayWindow_Rate = {-1, 20, 50, 100, 500};

		//double[] decayWindow_Rate = {-1};
		
		//double[] decayWindow_Rate = {100, 200, 400, 800};

		//Globals.setAdaptiveControl("None");

		//double[] mags = {0.2,0.4};
		//double[] mags = { 0.25,0.75 };
		//double[] mags = {0.1, 0.001, 0.000001};
		
		//double[] decayWindow_Rate = {-1, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005,  0.000001};
		
		//double[] decayWindow_Rate = {-1, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001};
		
		//double[] decayWindow_Rate = {-1, 0.1, 0.01, 0.005, 0.00005, 0.00001};
		//double[] decayWindow_Rate = {-1, 20, 50, 100, 500};
		
		//double[] decayWindow_Rate = {-1, 0.1, 0.01, 0.005};
		
		//double[] decayWindow_Rate = {0.15, 0.05, 0.005};
		
		double[] decayWindow_Rate = {20, 50, 500};
		
		//double[] mags = {5E-3, 7.5E-4, 5E-4, 1E-4, 1E-5};
		//double[] mags = {5E-3, 1E-3, 5E-4, 1E-4, 1E-5};
		//double[] mags = {100, 200, 400, 800};

		Model learner = null;

		Instances structure = null;

		File sourceFile = null;

		/* ----------------------------------- */
		/* Find number of samples for plotting */
		/* ----------------------------------- */

		double[][][][] res = null;
		double[][][][] resError = null;
		
		int nSamplesForChart = 0;
		String data = "";
		String dw_name = "";
		double driftMagnitude = 0;

		int numCycles = 1;

		if (Globals.isGenerateDriftData()) {

			nSamplesForChart = (Globals.getTotalNInstancesBeforeDrift() + Globals.getTotalNInstancesDuringDrift()  + Globals.getTotalNInstancesAfterDrift()) / Globals.getPrequentialBufferOutputResolution();

			if ((Globals.getTotalNInstancesBeforeDrift() + Globals.getTotalNInstancesDuringDrift()  + Globals.getTotalNInstancesAfterDrift()) % Globals.getPrequentialBufferOutputResolution() !=0)
				nSamplesForChart++;

			res = new double[Globals.getNumExp()][decayWindow_Rate.length][numFlows][nSamplesForChart];
			resError = new double[Globals.getNumExp()][decayWindow_Rate.length][numFlows][nSamplesForChart];

		} else {

			// dataset is provided, assuming it already has drift 

			data = Globals.getTrainFile();
			if (data.isEmpty()) {
				System.err.println("evaluation: No Training File given");
				System.exit(-1);
			}

			sourceFile = new File(data);
			if (!sourceFile.exists()) {
				System.err.println("Train evaluation: File " + data + " not found!");
				System.exit(-1);
			}

			Globals.setSOURCEFILE(sourceFile);

			if (!Globals.isNumInstancesKnown()) {
				Globals.setNumberInstances(SUtils.determineNumData());
			}

			structure = SUtils.setStructure();

			int  N = (int) Globals.getNumberInstances();
			N *= numCycles;

			nSamplesForChart = N / Globals.getPrequentialBufferOutputResolution();

			if (N % Globals.getPrequentialBufferOutputResolution() !=0)
				nSamplesForChart++;

			res = new double[Globals.getNumExp()][decayWindow_Rate.length][numFlows][nSamplesForChart];
			resError = new double[Globals.getNumExp()][decayWindow_Rate.length][numFlows][nSamplesForChart];
		}

		/* ------------------------------------- */
		/* Chop-chop with the normal training   */
		/* ------------------------------------- */

		for (int exp = 0; exp < Globals.getNumExp(); exp++) {

			if (Globals.isVerbose()) {
				System.out.println("-------------------------------------------------------------");
				System.out.println("Experiment No. " + exp);
				System.out.println("-------------------------------------------------------------");
			}

			for (int d = 0; d < decayWindow_Rate.length; d++) {

				if (decayWindow_Rate[d] == -1) {
					Globals.setAdaptiveControl("None");
				} else {
					
					Globals.setAdaptiveControl("window");
					dw_name = "window=";
					
					//Globals.setAdaptiveControl("decay");
					//dw_name = "decay=";
					
					Globals.setAdaptiveControlParameter(decayWindow_Rate[d]);
				}

				if (Globals.isVerbose()) {
					//System.out.println("Magnitude. " + m);
					System.out.println(" --------------> Decay. " + decayWindow_Rate[d] + " <-------------- ");
				}

				// ---------------------------------------------------------------------------------------------
				// Start -- Generate Data
				// ---------------------------------------------------------------------------------------------
				if (Globals.isGenerateDriftData()) {

					//double magnitude = 0.4;
					System.out.println("Drift Type = " + Globals.getDriftType());
					driftMagnitude = Globals.getDriftMagnitude();
					System.out.println("Drfit Magnitude = " + driftMagnitude);

					if (Globals.getDriftType().equalsIgnoreCase("simplest")) {

						//sourceFile = Sampler.generateSimpleDrift(exp, driftMagnitude);			

						System.out.println("Calling TAN Drift generator");
						sourceFile = Sampler.generateTANDrift(exp, 0);

					} else if (Globals.getDriftType().equalsIgnoreCase("simplestKDB")) {

						System.out.println("Calling KDB (K=2) Drift generator");
						sourceFile = Sampler.generateKDBDrift(exp, 0.0);		

					} else if (Globals.getDriftType().equalsIgnoreCase("noDrift")) {

						sourceFile = Sampler.generateNoDrift(exp, 0.0);		

					} else if (Globals.getDriftType().equalsIgnoreCase("gBayesian")) {

						/* Gradual Drift Bayesian */
						sourceFile = Sampler.generateDriftGradualBayesian(exp, driftMagnitude);

					} else if (Globals.getDriftType().equalsIgnoreCase("gLR")) {

						/* Gradual Drift LR */
						sourceFile = Sampler.generateDriftGradual(exp, driftMagnitude);

					} else if (Globals.getDriftType().equalsIgnoreCase("abrupt")) {

						/* Abrupt Drift */
						sourceFile = Sampler.generateDriftData(exp, driftMagnitude);

					} else if (Globals.getDriftType().equalsIgnoreCase("withBayesError")) {

						/* New, where there is Bayes error present during the drift */
						sourceFile = Sampler.generateDriftGradualSwappingGenerator(exp, driftMagnitude);	

					}

					int numNoiseColumns = Globals.getNumRandAttributes();
					sourceFile = SUtils.addNoise(numNoiseColumns, sourceFile);

				} else {

					sourceFile = new File(data);
					if (!sourceFile.exists()) {
						System.err.println("Train evaluation: File " + data + " not found!");
						System.exit(-1);
					}

					Globals.setSOURCEFILE(sourceFile);
					if (!Globals.isNumInstancesKnown()) {
						Globals.setNumberInstances(SUtils.determineNumData());
					}

					// Add randomization here
					sourceFile = Sampler.generateSimpleDriftFromData(exp, numCycles);

				}
				// ---------------------------------------------------------------------------------------------
				// End -- Generate Data
				// ---------------------------------------------------------------------------------------------


				Globals.setSOURCEFILE(sourceFile);

				if (!Globals.isNumInstancesKnown()) {
					Globals.setNumberInstances(SUtils.determineNumData());
				}

				structure = SUtils.setStructure();

				double N = Globals.getNumberInstances();
				int nc = Globals.getNumClasses();
				System.out.println("<num data points, num classes> = <" + N + ", " + nc + ">");

				/* 
				 * Done with data generation, normal flow-prequential learning starts here
				 */
				for (int f = 0; f < numFlows; f++) {

					if (Globals.isVerbose()) {
						//System.out.println("Level (AnDE). " + f);
						System.out.println("Level (" + Globals.getModel() + "): " + f);
					}

					if (flowVal.equalsIgnoreCase("adaptiveControlParameter")) {

						Globals.setAdaptiveControlParameter(flowValues[f]);

					} else if (flowVal.equalsIgnoreCase("sgdTuningParameter")) {

						Globals.setSgdTuningParameter(flowValues[f]);

					} else if (flowVal.equalsIgnoreCase("lambda")) {

						Globals.setLambda(flowValues[f]);

					} else if (flowVal.equalsIgnoreCase("level")) {

						Globals.setLevel((int) flowValues[f]);

					}  

					String val = Globals.getModel();

					if (val.equalsIgnoreCase("AnDE")) {

						learner = new ande();
						learner.buildClassifier();

					} 

					ArffReader reader = new ArffReader(new BufferedReader(new FileReader(Globals.getSOURCEFILE()), Globals.getBUFFER_SIZE()), Globals.getARFF_BUFFER_SIZE());
					Instance row = null;

					int nErrors = 0;
					int totalTested = 0;
					int indexPlot = 0;
					int lineNo = 0;
					double rmse = 0;

					while ((row = reader.readInstance(structure)) != null) {

						double[] probs = learner.distributionForInstance(row);
						
						//System.out.println(Arrays.toString(probs));
						
						double[] results = SUtils.getResults(probs, (int) row.classValue(), probs.length);

						if (results[1] == 1) {
							nErrors++;
						}
						totalTested++;
						rmse += results[0];

						if (lineNo % Globals.getPrequentialBufferOutputResolution() == 0) {

							//if (Globals.isPlotRMSEResuts()) {
								res[exp][d][f][indexPlot] = 1.0 * rmse / totalTested;
								resError[exp][d][f][indexPlot] = 1.0 * nErrors / totalTested;
							//} else {
							//	res[exp][d][f][indexPlot] = 1.0 * nErrors / totalTested;
							//}

							indexPlot++;
							nErrors = 0;
							rmse = 0;
							totalTested = 0;
						}

						learner.update(row);

						lineNo++;
					}

					System.out.println("\nExp: " + exp + ". Read: " + lineNo + " data points, out of which learner was evaluated on: " + indexPlot);
					System.out.println();

				} // ends flows

				SUtils.deleteSourceFile();

			}// ends decayRates

		} // ends exp

		double[] minMean = new double[numFlows];
		Arrays.fill(minMean, Double.MAX_VALUE);

		// all experiments done; now averaging for all exps
		double[][][] meanErrorPlot = new double[numFlows][decayWindow_Rate.length][nSamplesForChart];
		double[][][] stdDevErrorPlot = new double[numFlows][decayWindow_Rate.length][nSamplesForChart];

		double[][][] meanErrorPlotError = new double[numFlows][decayWindow_Rate.length][nSamplesForChart];
		double[][][] stdDevErrorPlotError = new double[numFlows][decayWindow_Rate.length][nSamplesForChart];
		
		for (int m = 0; m < decayWindow_Rate.length; m++) { // decays
			for (int c = 0; c < numFlows; c++) { // classifiers
				for (int indexPlot = 0; indexPlot < nSamplesForChart; indexPlot++) {
					// compute mean error
					for (int exp = 0; exp < Globals.getNumExp(); exp++) {
						//note I remove bayes optimal from error value
						// if want everything, then remove the second term
						//meanErrorPlot[c][m][indexPlot] += res[exp][m][c][indexPlot]-res[exp][m][numFlows-1][indexPlot];
						
						meanErrorPlot[c][m][indexPlot] += res[exp][m][c][indexPlot];
						meanErrorPlotError[c][m][indexPlot] += resError[exp][m][c][indexPlot];
					}
					
					meanErrorPlot[c][m][indexPlot] /= Globals.getNumExp();
					meanErrorPlotError[c][m][indexPlot] /= Globals.getNumExp();
					
					// compute stddev error
					for (int exp = 0; exp < Globals.getNumExp(); exp++) {
						//note I remove bayes optimal from error value
						// if want everything, then remove the second term
						//double diff = (res[exp][m][c][indexPlot]-res[exp][m][numFlows-1][indexPlot] - meanErrorPlot[c][m][indexPlot]);
						
						double diff1 = (res[exp][m][c][indexPlot] - meanErrorPlot[c][m][indexPlot]);
						double diff2 = (resError[exp][m][c][indexPlot] - meanErrorPlotError[c][m][indexPlot]);
						
						stdDevErrorPlot[c][m][indexPlot] += diff1 * diff1;
						stdDevErrorPlotError[c][m][indexPlot] += diff2 * diff2;
					}
					
					stdDevErrorPlot[c][m][indexPlot] /= Globals.getNumExp();
					stdDevErrorPlotError[c][m][indexPlot] /= Globals.getNumExp();
					
					stdDevErrorPlot[c][m][indexPlot] = Math.sqrt(stdDevErrorPlot[c][m][indexPlot]);
					stdDevErrorPlotError[c][m][indexPlot] = Math.sqrt(stdDevErrorPlotError[c][m][indexPlot]);
					
					minMean[c] = Math.min(minMean[c], meanErrorPlot[c][m][indexPlot]);
				}
			}
		}
		
		String fileName = Globals.getOuputResultsDirectory();
		if (!new File(fileName).exists()) {
			new File(fileName).mkdirs();
		}
		//fileName += evaluationDrift.class.getSimpleName();
		//fileName += "-nAttributes=" + Globals.getDriftNAttributes();
		fileName += "-nExp=" + Globals.getNumExp();
		//fileName += "-before=" + Globals.getTotalNInstancesBeforeDrift();
		//fileName += "-after=" + Globals.getTotalNInstancesAfterDrift();
		fileName += "-drfitMagnitude=" + Globals.getDriftMagnitude();
		fileName += "-drfitDelta=" + Globals.getDriftDelta();
		fileName += "-drfitMagnitude2=" + Globals.getDriftMagnitude2();
		fileName += "-drfitMagnitude3=" + Globals.getDriftMagnitude3();
		fileName += "-drfitType=" + Globals.getDriftType();
		
		//if (Globals.isPlotRMSEResuts()) {
		//	fileName += "-RMSE";
		//} else {
		//	fileName += "-01Loss";
		//}
		
		fileName += "-" + System.currentTimeMillis() / 1000;
				
		for (int c = 0; c < numFlows; c++) {
			
			String f1 = fileName + "-RMSE" + "_Level" + flowValues[c] + ".results";
			String f2 = fileName + "-01Loss" + "_Level" + flowValues[c] + ".results";
			
			File out1 = new File(f1);
			File out2 = new File(f2);
			
			Writer w1 = new BufferedWriter(new FileWriter(out1));
			Writer w2 = new BufferedWriter(new FileWriter(out2));
			
			for (int indexPlot = 0; indexPlot < nSamplesForChart; indexPlot++) {
				for (int m = 0; m < decayWindow_Rate.length; m++) { // decays
					double sum1 = 0;
					double sum2 = 0;
					for (int exp = 0; exp < Globals.getNumExp(); exp++) {
						sum1 += res[exp][m][c][indexPlot];
						sum2 += resError[exp][m][c][indexPlot];
					}
					sum1 /= Globals.getNumExp();
					sum2 /= Globals.getNumExp();
					
					w1.write(sum1 + ", ");
					w2.write(sum2 + ", ");
				}
				w1.write("\n");
				w2.write("\n");
			}
			
			w1.close();
			w2.close();
		}
		
		res = null;
		resError = null;

		//Plotting.plot(fileName, numFlows, flowValues, driftMagnitude, decayWindow_Rate, dw_name, nSamplesForChart, meanErrorPlot, stdDevErrorPlot, minMean);
		//Plotting.plot2(fileName, numFlows, flowValues, driftMagnitude, decayWindow_Rate, dw_name, nSamplesForChart, meanErrorPlot, stdDevErrorPlot, minMean);
		
		Plotting.plot(fileName, numFlows, flowValues, driftMagnitude, decayWindow_Rate, dw_name, nSamplesForChart, meanErrorPlotError, stdDevErrorPlot, minMean);
		Plotting.plot2(fileName, numFlows, flowValues, driftMagnitude, decayWindow_Rate, dw_name, nSamplesForChart, meanErrorPlotError, stdDevErrorPlot, minMean);
		
	}

}
