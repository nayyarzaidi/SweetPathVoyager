package evaluation;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Map.Entry;

import org.apache.commons.math3.random.MersenneTwister;

import utils.Globals;
import utils.SUtils;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffSaver;
import weka.core.converters.Saver;
import weka.core.converters.ArffLoader.ArffReader;
import model.Model;
import model.ande.ande;

public class evaluationPreprocess {

	public static void learn() throws Exception {

		Globals.printWelcomeMessageWranglerini();

		String data = Globals.getTrainFile();

		if (data.isEmpty()) {
			System.err.println("evaluationPreprocess: No Training File given");
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


		if (Globals.getPreProcessParameter().equalsIgnoreCase("Explore")) {

			System.out.println("------------------------");
			System.out.println("Explore");
			System.out.println("-- Prints distinct values of each categorical attribute");
			System.out.println("-- Prints range of each numeric attributes");
			System.out.println("\n");
			System.out.println("E.g., usage:  java -Xmx32G -cp aa.jar -trainFile  abc.arff -experimentType preprocess -preProcessParameter Explore -ignoreAttributes {} -attributeType {0,1,1,1,1,1,1,1,1,1,1,1} -tempDirectory /Users/nayyar/Desktop/AA/temp2/ -ouputResultsDirectory /Users/nayyar/Desktop/AA/temp2/ ");
			System.out.println("------------------------");

			System.out.println("-- Attributes to be ignored: " + Globals.getIgnoreAttributes());
			System.out.println("-- Attribute Type: " + Globals.getAttributeType());

			boolean[] isNumericFlag = SUtils.getBooleanFromLine(Globals.getAttributeType());	
			int n = isNumericFlag.length;

			BufferedReader reader = new BufferedReader(new FileReader(sourceFileTrain), Globals.getBUFFER_SIZE());

			int numDiscrete = 0;
			for (int u = 0; u < n; u++) {
				if (!isNumericFlag[u]) {
					numDiscrete++;
				}
			}

			if (Globals.isVerbose()) {
				System.out.println("-- Attribute Type: " + Arrays.toString(isNumericFlag));
				System.out.println("-- Number of attributes according to the header info (-attributeType) is: " + n);
				System.out.println("-- Reading file, First Pass: Extracting distinct attribute-values");
				System.out.println("-- Number of discrete Attributes: " + numDiscrete);
			}

			//HashMap<String, Integer> map = new HashMap<String, Integer>();
			List<Map<String, Integer>> listOfMaps = new ArrayList<Map<String, Integer>>();
			HashMap<String, Integer> map;
			for (int i = 0; i < n; i++) {
				map = new HashMap<String, Integer>();
				listOfMaps.add(map);
			}

			ArrayList<ArrayList<Double>> listOfDoubles = new ArrayList<ArrayList<Double>>();
			ArrayList<Double> doub;
			for (int i = 0; i < n; i++) {
				doub = new ArrayList<Double>();
				listOfDoubles.add(doub);
			}

			int[] ignoreAttributes = SUtils.getIntegerFromLine(Globals.getIgnoreAttributes(), ',');

			String line = null;
			int N = 0;
			int[] numberMissing = new int[n];

			while ((line = reader.readLine()) != null) {

				String[] vals = SUtils.getStringFromLine(line, ',');

				if (vals.length != n) {
					System.out.println("Header and data do not comply");
					System.exit(-1);
				}

				for (int u = 0; u < n; u++) {
					if (!SUtils.inArray(u, ignoreAttributes)) {
						if (!isNumericFlag[u]) {
							// Discrete Attribute
							if (vals[u].equals("")) {
								listOfMaps.get(u).put("Missing",u);
								numberMissing[u]++;
							} else {
								listOfMaps.get(u).put(vals[u],u);
							}
						} else {
							// Numeric Attribute
							if (vals[u].equals("")) {
								numberMissing[u]++;
							} else {
								listOfDoubles.get(u).add(Double.parseDouble(vals[u]));
							}
						}
					}
				}
				N++;
			}

			double[][] statistics = new double[n][5];
			for (int u = 0; u < n; u++) {
				if (isNumericFlag[u]) {
					ArrayList<Double> list = listOfDoubles.get(u);
					statistics[u][0] = SUtils.getArrayListDoubleMean(list);
					statistics[u][1] = SUtils.getArrayListDoubleVariance(list);
					statistics[u][2] = SUtils.getArrayListDoubleMin(list);
					statistics[u][3] = SUtils.getArrayListDoubleMax(list);
					statistics[u][4] = SUtils.getArrayListDoubleMode(list);
				}
			}

			System.out.println("Done with the First Pass \n");

			System.out.println("Read a total of " + N + " records");
			for (int u = 0; u < n; u++) {
				if (!SUtils.inArray(u, ignoreAttributes)) {
					if (!isNumericFlag[u]) {
						System.out.println("Attribute " + u + " is Discrete");
						System.out.println("No. of values:  " + listOfMaps.get(u).entrySet().size() + " values.");
						System.out.println("Number of Missing values: " + numberMissing[u]);
					} else {
						System.out.println("Attribute " + u + " is Numeric.");
						System.out.println("Mean:  " + statistics[u][0] + ", Variance: " + statistics[u][1] + ", Min-val: " + statistics[u][2] + ", Max-val: " + statistics[u][3] + ", Mode: " + + statistics[u][4]);
						System.out.println("Number of Missing values: " + numberMissing[u]);
					}
				} else {
					System.out.println("Attribute " + u + " Ignored.");
				}
			}

		} else if (Globals.getPreProcessParameter().equalsIgnoreCase("CreateHeader")) {

			System.out.println("------------------------");
			System.out.println("CreateHeader");
			System.out.println("-- Takes in raw CSV file, and adds header to it and write it back to the file");
			System.out.println("\n");
			System.out.println("E.g., usage:  java -Xmx32G -cp aa.jar -trainFile  abc.arff -experimentType preprocess -preProcessParameter CreateHeader -classAttribute 14 -ignoreAttributes {0,1,2} -attributeType {0,1,1,1,1,1,1,1,1,1,1,1} -tempDirectory /Users/nayyar/Desktop/AA/temp2/ -ouputResultsDirectory /Users/nayyar/Desktop/AA/temp2/ ");
			System.out.println("------------------------");

			System.out.println("Attributes to be ignored: " + Globals.getIgnoreAttributes());
			System.out.println("Class Attribute: " + Globals.getClassAttribute());
			System.out.println("Attribute Type: " + Globals.getAttributeType());

			boolean[] isNumericFlag = SUtils.getBooleanFromLine(Globals.getAttributeType());
			int n = isNumericFlag.length;
			int classAttribute = Globals.getClassAttribute();

			if (classAttribute == -1) {
				System.out.println("No Class attribute is provided, will add an extra attribute at the end with ? and header {0, 1} -- Please modify the header yourself");
			}

			BufferedReader reader = new BufferedReader(new FileReader(sourceFileTrain), Globals.getBUFFER_SIZE());

			int numDiscrete = 0;
			for (int u = 0; u < n; u++) {
				if (!isNumericFlag[u]) {
					numDiscrete++;
				}
			}

			if (Globals.isVerbose()) {
				System.out.println("-- Attribute Type: " + Arrays.toString(isNumericFlag));
				System.out.println("-- Number of attributes according to the header info (-attributeType) is: " + n);
				System.out.println("-- Reading file, First Pass: Extracting distinct attribute-values");
				System.out.println("-- Number of discrete Attributes: " + numDiscrete);
			}

			//HashMap<String, Integer> map = new HashMap<String, Integer>();
			List<Map<String, Integer>> listOfMaps = new ArrayList<Map<String, Integer>>();
			HashMap<String, Integer> map;
			for (int i = 0; i < n; i++) {
				map = new HashMap<String, Integer>();
				listOfMaps.add(map);
			}

			int[] ignoreAttributes = SUtils.getIntegerFromLine(Globals.getIgnoreAttributes(), ',');

			String line = null;
			int index = 1;
			int N = 0;
			while ((line = reader.readLine()) != null) {

				String[] vals = SUtils.getStringFromLine(line, ',');

				if (vals.length != n) {
					System.out.println("Header and data do not comply");
					System.exit(-1);
				}

				for (int u = 0; u < n; u++) {
					if (!SUtils.inArray(u, ignoreAttributes)) {
						if (!isNumericFlag[u]) {
							if (vals[u].equals("")) {
								listOfMaps.get(u).put("Missing",u);
							} else {
								listOfMaps.get(u).put(vals[u],u);
							}
						}
					}
				}

				N++;
			}

			if (Globals.isVerbose()) {
				System.out.println("-- Done with the First Pass \n");
				System.out.println("Read a total of " + N + " records");
			}

			for (int u = 0; u < n; u++) {
				if (!SUtils.inArray(u, ignoreAttributes)) {

					if (!isNumericFlag[u]) {
						System.out.println("Attribute " + u + " is Discrete and takes " + listOfMaps.get(u).entrySet().size() + " values.");
					} else {
						System.out.println("Attribute " + u + " is Numeric.");
					}

				} else {
					System.out.println("Attribute " + u + " Ignored.");
				}

			}

			if (Globals.isVerbose()) {
				System.out.println("-- Starting Second Pass \n");
			}

			String outputFileName = Globals.getTempDirectory() + "/" + Globals.getDataSetName() + "_CreateHeader.arff"; 
			File out = new File(outputFileName);

			Writer w = new BufferedWriter(new FileWriter(out));

			String header = "";
			header += "@relation '" + outputFileName + "'\n\n";
			w.write(header);
			int numAtt = 1; 
			int discAtt = 1;
			for (int u = 0; u < n; u++) {
				if (!SUtils.inArray(u, ignoreAttributes)  && u != classAttribute) {
					if (!isNumericFlag[u]) {

						System.out.println("-- Processing dX Attribute" + u);

						/* Discrete Attribute */
						w.write("@attribute dX" + discAtt + " { ");
						int v = 0;
						for (Entry<String, Integer> entry : listOfMaps.get(u).entrySet()) {
							if (v == listOfMaps.get(u).entrySet().size() - 1) {  
								w.write(entry.getKey());
							} else {
								w.write(entry.getKey() + ", ");
							}
							if (v % 100000 == 0)
								System.out.print(".");
							v++;
						}
						w.write(" }\n");
						discAtt++;
					} else {
						/* Numeric Attribute */
						w.write("@attribute nX" + numAtt + " real \n");
						numAtt++;
					}
				}
			}

			w.write("@attribute class { ");
			if (classAttribute == -1) {
				w.write("0 , 1");
			} else {
				int v = 0;
				for (Entry<String, Integer> entry : listOfMaps.get(classAttribute).entrySet()) {
					if (v == listOfMaps.get(classAttribute).entrySet().size() - 1)   {
						w.write(entry.getKey());
					} else {
						w.write(entry.getKey() + ", ");
					}
					v++;
				}
			}
			w.write(" }\n");

			w.write("\n@data\n\n");

			if (Globals.isVerbose()) {
				System.out.println("-- Done with Writing the header");
			}

			reader = new BufferedReader(new FileReader(sourceFileTrain), Globals.getBUFFER_SIZE());
			int i = 0;
			while ((line = reader.readLine()) != null) {

				if (i%100000 == 0) {
					System.out.print(".");
				}

				String[] vals = SUtils.getStringFromLine(line, ',');

				for (int u = 0; u < n; u++) {
					if (!SUtils.inArray(u, ignoreAttributes) && u != classAttribute) {

						if (!isNumericFlag[u]) {
							if (vals[u].equals("")) {
								w.write("Missing" + ", ");
							} else {
								w.write(vals[u] + ", ");
							}
						} else {
							if (vals[u].equals("")) {
								w.write("?" + ", ");
							} else {
								w.write(vals[u] + ", ");
							}
						}

					}
				}

				if (classAttribute == -1) {
					w.write("?");
				} else {
					w.write(vals[classAttribute]);
				}

				w.write("\n");
				i++;
			}

			w.close();
			System.out.println("Header Successfully attached - Thoroughly inspect before use " + out);


		} else if (Globals.getPreProcessParameter().equalsIgnoreCase("Discretize")) {

			/* --------------------------------------------------------------------------
			 * Discretized data already in Arff format and write the discretized file to the 
			 * disk.
			 * -------------------------------------------------------------------------- */

			System.out.println("------------------------");
			System.out.println("Discretize");
			System.out.println("-- Takes data in ARFF format and discretize it");
			System.out.println("\n");
			System.out.println("E.g., usage:  java -Xmx32G -cp aa.jar -trainFile  abc.arff -experimentType preprocess -preProcessParameter Discretize -classAttribute 14 -ignoreAttributes {0,1,2} -attributeType {0,1,1,1,1,1,1,1,1,1,1,1} [-discretizeOutOfCore] -tempDirectory /Users/nayyar/Desktop/AA/temp2/ -ouputResultsDirectory /Users/nayyar/Desktop/AA/temp2/ ");
			System.out.println("------------------------");
			
			Globals.setSOURCEFILE(sourceFileTrain);

			Instances structure = SUtils.setStructure();

			int N = (int) Globals.getNumberInstances();
			int nc = Globals.getNumClasses();
			int n = Globals.getNumAttributes();

			System.out.println("<num data points, num classes> = <" + N + ", " + n + ", " + nc + ">");

			for (int u = 0; u < n; u++) {
				String val = (Globals.getIsNumericTrue()[u]) ? "Numeric" : "Discrete";
				System.out.println("Attribute " + u + ": " + val + " -- ParamsPerAtt: " + Globals.getParamsPerAtt()[u]);
			}

			System.out.println("Class Attribute: " + Globals.getClassAttribute());

			int[] ignoreAttributes = new int[n];
			for (int u = 0; u < n; u++) {
				ignoreAttributes[u] = u;
			}

			File[] discreteFiles = new File[n];

			if (Globals.isDiscretizeOutOfCore()) {

				/* ------------------------------------------------------------------- */
				/* Discretize out of core, file is too big to be loaded into main memory */
				/* ------------------------------------------------------------------- */

				for (int u = 0; u < n; u++) {
					if (structure.attribute(u).isNumeric()) {
						ignoreAttributes[u] = -1;

						String outputFileName = Globals.getTempDirectory() + "/" + Globals.getDataSetName() + "_Sliced_Att" + u + ".arff"; 

						File out = sliceDataSet(sourceFileTrain, ignoreAttributes, outputFileName);

						/* In-core discretize out */
						outputFileName = Globals.getTempDirectory() + "/" + Globals.getDataSetName() + "_Discretized_Sliced_Att" + u +".arff";
						File outdiscrete = discretizeDatSet(out, outputFileName);

						discreteFiles[u] = outdiscrete;

						// set ignoreAttributes to way it was
						ignoreAttributes[u] = u;
						out.delete();
					}
				}

				/* Okay files are written, now chop-chop to collate the files into one */
				System.out.println("Now Collating results from all the Discretized Files");

				String outputFileName = Globals.getTempDirectory() + "/" + Globals.getDataSetName() + "_DiscretizedOOC.arff"; 

				File out = collateDiscretizedFiles(sourceFileTrain, outputFileName, discreteFiles);

			} else {

				/* ------------------------------------------------------------------- */
				/* Discretize in-core */
				/* ------------------------------------------------------------------- */

				String outputFileName = Globals.getTempDirectory() + "/" + Globals.getDataSetName() + "_Discretized.arff"; 

				File out = discretizeDatSet(sourceFileTrain, outputFileName);

			}

		} else if (Globals.getPreProcessParameter().equalsIgnoreCase("Slice")) {

			/* --------------------------------------------------------------------------
			 * Slice the data: Ignore attributes in ignore list
			 * -------------------------------------------------------------------------- */

			System.out.println("------------------------");
			System.out.println("Slice");
			System.out.println("------------------------");

			int[] ignoreAttributes = SUtils.getIntegerFromLine(Globals.getIgnoreAttributes(), ',');

			String outputFileName = Globals.getTempDirectory() + "/" + Globals.getDataSetName() + "_Sliced.arff"; 

			File out = sliceDataSet(sourceFileTrain, ignoreAttributes, outputFileName);


		} else if (Globals.getPreProcessParameter().equalsIgnoreCase("Dice")) {

			System.out.println("------------------------");
			System.out.println("Diced");
			System.out.println("------------------------");

			ArffReader readerTrain = new ArffReader(new BufferedReader(new FileReader(Globals.getTrainFile()), Globals.getBUFFER_SIZE()), Globals.getARFF_BUFFER_SIZE());
			Instances structureTrain = readerTrain.getStructure();
			structureTrain.setClassIndex(structureTrain.numAttributes() - 1);

			if (Globals.getDicedAt() == 0) {

				if (Globals.isVerbose()) {
					System.out.println("Diced Perecentage = " + Globals.getDicedPercentage());
					System.out.println("Diced Stratification = " + Globals.isDicedStratified());
				}

				if (Globals.isDicedStratified()) {
					/* go through the data once to determine the class distribution */
					BitSet res = SUtils.getStratifiedIndices((int)Globals.getDicedPercentage(), Globals.getTrainFile());

					/* Now sample the dataset */
					String outputFileNameTrain = Globals.getTempDirectory() + "/" + Globals.getDataSetName() + "_Train.arff"; 
					File outTrain = new File(outputFileNameTrain);
					Writer wTrain = new BufferedWriter(new FileWriter(outTrain));

					wTrain.write(structureTrain.toString());

					String outputFileNameTest = Globals.getTempDirectory() + "/" + Globals.getDataSetName() + "_Test.arff"; 
					File outTest = new File(outputFileNameTest);
					Writer wTest = new BufferedWriter(new FileWriter(outTest));

					wTest.write(structureTrain.toString());

					Instance current = null;
					int N = 0;
					int Ntrain = 0;
					int Ntest = 0;
					while ((current = readerTrain.readInstance(structureTrain)) != null) {
						if (!res.get(N)) {
							wTrain.write(current.toString());
							wTrain.write("\n");
							Ntrain++;
						} else {
							wTest.write(current.toString());
							wTest.write("\n");
							Ntest++;
						}
						N++;
					}

					wTrain.close();
					wTest.close();

					if (Globals.isVerbose()) {
						System.out.println("No. of training data points = " + Ntrain);
						System.out.println("No. of testing data points = " + Ntest);
					}

					System.out.println("Data Successfully Diced " + outTrain + ", and " + outTest);

				} else {

					/* Now sample the dataset */
					String outputFileNameTrain = Globals.getTempDirectory() + "/" + Globals.getDataSetName() + "_Train.arff"; 
					File outTrain = new File(outputFileNameTrain);
					Writer wTrain = new BufferedWriter(new FileWriter(outTrain));

					wTrain.write(structureTrain.toString());

					String outputFileNameTest = Globals.getTempDirectory() + "/" + Globals.getDataSetName() + "_Test.arff"; 
					File outTest = new File(outputFileNameTest);
					Writer wTest = new BufferedWriter(new FileWriter(outTest));

					wTest.write(structureTrain.toString());

					Instance current = null;
					int N = 0;
					int Ntrain = 0;
					int Ntest = 0;
					while ((current = readerTrain.readInstance(structureTrain)) != null) {
						if ((N % 2) == 0) {
							wTrain.write(current.toString());
							wTrain.write("\n");
							Ntrain++;
						} else {
							wTest.write(current.toString());
							wTest.write("\n");
							Ntest++;
						}
						N++;
					}

					wTrain.close();
					wTest.close();

					if (Globals.isVerbose()) {
						System.out.println("No. of training data points = " + Ntrain);
						System.out.println("No. of testing data points = " + Ntest);
					}

					System.out.println("Data Successfully Diced " + outTrain + ", and " + outTest);

				}
			} else {
				/* Split into two files based on dicedAt parameter */

				System.out.println("Will be dicing at: " + Globals.getDicedAt());

				String outputFileNameTrain = Globals.getTempDirectory() + "/" + Globals.getDataSetName() + "_Train.arff"; 
				File outTrain = new File(outputFileNameTrain);
				Writer wTrain = new BufferedWriter(new FileWriter(outTrain));

				wTrain.write(structureTrain.toString());

				String outputFileNameTest = Globals.getTempDirectory() + "/" + Globals.getDataSetName() + "_Test.arff"; 
				File outTest = new File(outputFileNameTest);
				Writer wTest = new BufferedWriter(new FileWriter(outTest));

				wTest.write(structureTrain.toString());

				Instance current = null;
				int N = 0;
				while ((current = readerTrain.readInstance(structureTrain)) != null) {
					if (N < Globals.getDicedAt()) {
						wTrain.write(current.toString());
						wTrain.write("\n");
					} else {
						wTest.write(current.toString());
						wTest.write("\n");
					}
					N++;
				}

				wTrain.close();
				wTest.close();

				System.out.println("Data Successfully Diced " + outTrain + ", and " + outTest);
			}

		} else if (Globals.getPreProcessParameter().equalsIgnoreCase("Normalize")) {

		} else if (Globals.getPreProcessParameter().equalsIgnoreCase("HeaderAlignment")) {

			System.out.println("------------------------");
			System.out.println("Header Alignment");
			System.out.println("------------------------");

			String dataTest = Globals.getTestFile();

			File sourceFileTest;
			sourceFileTest = new File(dataTest);
			if (!sourceFileTest.exists()) {
				System.err.println("Test evaluation: File " + data + " not found!");
				System.exit(-1);
			}

			if (Globals.isVerbose()) {
				System.out.println("Testing Source File is at: " + sourceFileTest.getAbsolutePath());
			}

			ArffReader readerTrain = new ArffReader(new BufferedReader(new FileReader(Globals.getTrainFile()), Globals.getBUFFER_SIZE()), Globals.getARFF_BUFFER_SIZE());
			Instances structureTrain = readerTrain.getStructure();
			structureTrain.setClassIndex(structureTrain.numAttributes() - 1);

			ArffReader readerTest = new ArffReader(new BufferedReader(new FileReader(Globals.getTestFile()), Globals.getBUFFER_SIZE()), Globals.getARFF_BUFFER_SIZE());
			Instances structureTest = readerTest.getStructure();
			structureTest.setClassIndex(structureTest.numAttributes() - 1);

			/* ---------------------------------------------------------------------------------- */
			/* Create one header from train and test */
			/* ---------------------------------------------------------------------------------- */

			List<Map<String, Integer>> listOfMaps = new ArrayList<Map<String, Integer>>();
			HashMap<String, Integer> map;
			for (int i = 0; i < structureTrain.numAttributes() - 1; i++) {
				map = new HashMap<String, Integer>();
				listOfMaps.add(map);
			}

			for (int u = 0; u < structureTrain.numAttributes() - 1; u++) {
				if (structureTrain.attribute(u).isNominal()) {
					for (int uval = 0; uval < structureTrain.attribute(u).numValues(); uval++) {
						listOfMaps.get(u).put(structureTrain.attribute(u).value(uval), u);
					}
				}
			}

			for (int u = 0; u < structureTest.numAttributes() - 1; u++) {
				if (structureTest.attribute(u).isNominal()) {
					for (int uval = 0; uval < structureTest.attribute(u).numValues(); uval++) {
						listOfMaps.get(u).put(structureTest.attribute(u).value(uval), u);
					}
				}
			}

			for (int u = 0; u < structureTrain.numAttributes() - 1; u++) {
				if (structureTrain.attribute(u).isNominal()) {
					System.out.println("Attribute " + u + " is Discrete and takes " + listOfMaps.get(u).entrySet().size() + " values.");
				} else {
					System.out.println("Attribute " + u + " is Numeric.");
				}
			}

			String outputFileName = Globals.getTempDirectory() + "/" + Globals.getDataSetName() + "_HeaderAlignment.arff"; 
			File out = new File(outputFileName);

			Writer w = new BufferedWriter(new FileWriter(out));

			String header = "";
			header += "@relation '" + outputFileName + "'\n\n";
			w.write(header);
			int numAtt = 1; 
			int discAtt = 1;
			for (int u = 0; u < structureTrain.numAttributes() - 1; u++) {
				if (structureTest.attribute(u).isNominal()) {

					System.out.println("Processing dX Attribute" + u);

					/* Discrete Attribute */
					w.write("@attribute dX" + discAtt + " { ");
					int v = 0;
					for (Entry<String, Integer> entry : listOfMaps.get(u).entrySet()) {
						if (v == listOfMaps.get(u).entrySet().size() - 1) {  
							w.write(entry.getKey());
						} else {
							w.write(entry.getKey() + ", ");
						}
						if (v % 100000 == 0)
							System.out.print(".");
						v++;
					}
					w.write(" }\n");
					discAtt++;
				} else {
					/* Numeric Attribute */
					w.write("@attribute nX" + numAtt + " real \n");
					numAtt++;
				}
			}

			w.write("@attribute class { ");

			int v = 0;
			for (int c = 0; c < structureTrain.numClasses(); c++) {
				if (v == structureTrain.numClasses() - 1)   {
					w.write(structureTrain.classAttribute().value(c));
				} else {
					w.write(structureTrain.classAttribute().value(c) + ", ");
				}
				v++;
			}

			w.write(" }\n");

			w.write("\n@data\n\n");

			System.out.println("Done with Writing the header");

			/* ---------------------------------------------------------------------------------- */
			/* Concatenate the two files */
			/* ---------------------------------------------------------------------------------- */

			Instance current = null;
			while ((current = readerTrain.readInstance(structureTrain)) != null) {
				w.write(current.toString());
				w.write("\n");
			}
			while ((current = readerTest.readInstance(structureTest)) != null) {
				w.write(current.toString());
				w.write("\n");
			}
			w.close();
			System.out.println("Data Successfully TargetAlignment (Phase 1) " + out);

		} else if (Globals.getPreProcessParameter().equalsIgnoreCase("MissingImputate")) {

			System.out.println("------------------------");
			System.out.println("Missing Imputate (Discretized)");
			System.out.println("------------------------");

			int classAttribute = Globals.getClassAttribute();

			Globals.setSOURCEFILE(sourceFileTrain);
			Instances structure = SUtils.setStructure();

			int N = (int) Globals.getNumberInstances();
			int nc = Globals.getNumClasses();
			int n = Globals.getNumAttributes();

			System.out.println("<num data points, num classes> = <" + N + ", " + n + ", " + nc + ">");

			for (int u = 0; u < n; u++) {
				String val = (Globals.getIsNumericTrue()[u]) ? "Numeric" : "Discrete";
				System.out.println("Attribute " + u + ": " + val + " -- ParamsPerAtt: " + Globals.getParamsPerAtt()[u]);
			}

			System.out.println("Class Attribute: " + Globals.getClassAttribute());

			boolean[] isMissing = new boolean[n];
			for (int i = 0; i < n; i++) {
				isMissing[i] = false;
			}

			ArffReader reader = new ArffReader(new BufferedReader(new FileReader(Globals.getTrainFile()), Globals.getBUFFER_SIZE()), Globals.getARFF_BUFFER_SIZE());
			structure = reader.getStructure();
			structure.setClassIndex(structure.numAttributes() - 1);

			Instance current = null;
			while ((current = reader.readInstance(structure)) != null) {

				for (int u = 0; u < n; u++) {
					if (u != classAttribute) {
						if (current.isMissing(u)) {
							isMissing[u] = true;;
						}
					}
				}
			}

			/* Re-write header */
			String outputFileName = Globals.getTempDirectory() + "/" + Globals.getDataSetName() + "_MissingImputated.arff"; 
			File out = new File(outputFileName);

			Writer w = new BufferedWriter(new FileWriter(out));

			String header = "";
			header += "@relation '" + outputFileName + "'\n\n";
			w.write(header);

			for (int u = 0; u < n; u++) {
				if (u != classAttribute) {
					w.write("@attribute " + structure.attribute(u).name() + " { ");
					for (int v = 0; v < structure.attribute(u).numValues(); v++) {
						if (v == structure.attribute(u).numValues() - 1) {  
							w.write(structure.attribute(u).value(v));
						} else {
							w.write(structure.attribute(u).value(v) + ", ");
						}
					}
					if (isMissing[u]) {
						w.write(", missing}\n");
					} else {
						w.write(" }\n");
					}
				}
			}

			w.write("@attribute class { ");
			if (classAttribute == -1) {
				w.write("0 , 1");
			} else {
				for (int v = 0; v < structure.attribute(classAttribute).numValues(); v++) {
					if (v == structure.attribute(classAttribute).numValues() - 1)   {
						w.write(structure.attribute(classAttribute).value(v));
					} else {
						w.write(structure.attribute(classAttribute).value(v) + ", ");
					}
				}
			}
			w.write(" }\n");

			w.write("\n@data\n\n");

			/* Re-write data */
			reader = new ArffReader(new BufferedReader(new FileReader(Globals.getTrainFile()), Globals.getBUFFER_SIZE()), Globals.getARFF_BUFFER_SIZE());
			structure = reader.getStructure();
			structure.setClassIndex(structure.numAttributes() - 1);

			current = null;
			while ((current = reader.readInstance(structure)) != null) {

				for (int u = 0; u < n; u++) {
					if (u != classAttribute) {
						if (!current.isMissing(u)) {
							w.write(current.attribute(u).value((int) current.value(u))+ ", ");
						} else {
							w.write("missing" + ",");
						}
					}
				}

				if (classAttribute == -1) {
					w.write("?");
				} else {
					if (!current.isMissing(classAttribute)) {
						w.write(current.attribute(classAttribute).value((int) current.value(classAttribute)) + "");
					} else {
						w.write("?");
					}
				}

				w.write("\n");
			} 

			w.close();
			System.out.println("Data Successfully MissingImputed (discretized) " + out);

		} else if (Globals.getPreProcessParameter().equalsIgnoreCase("Binarize")) {

			System.out.println("------------------------");
			System.out.println("Binarize");
			System.out.println("Assumption 1: all attributes are already discretized");
			System.out.println("Assumption 2: class is the last attribute");
			System.out.println("------------------------");

			Globals.setSOURCEFILE(sourceFileTrain);
			Instances structure = SUtils.setStructure();

			int N = (int) Globals.getNumberInstances();
			int nc = Globals.getNumClasses();
			int n = Globals.getNumAttributes();

			System.out.println("<num data points, num classes> = <" + N + ", " + n + ", " + nc + ">");

			/* Re-write header */
			String outputFileName = Globals.getTempDirectory() + "/" + Globals.getDataSetName() + "_Binarized.arff"; 
			File out = new File(outputFileName);

			Writer w = new BufferedWriter(new FileWriter(out));

			String header = "";
			header += "@relation '" + outputFileName + "'\n\n";
			w.write(header);

			for (int u = 0; u < n; u++) {
				for (int uval = 0; uval < Globals.getParamsPerAtt()[u]; uval++) {
					w.write("@attribute att_" + structure.attribute(u).name()  + "_" + u + "_val_" + uval  + " { 0, 1 } \n");
				}
			}

			w.write("@attribute class { ");
			for (int v = 0; v < structure.attribute(n).numValues(); v++) {
				if (v == structure.attribute(n).numValues() - 1)   {
					w.write(structure.attribute(n).value(v));
				} else {
					w.write(structure.attribute(n).value(v) + ", ");
				}
			}

			w.write(" }\n");

			w.write("\n@data\n\n");

			ArffReader reader = new ArffReader(new BufferedReader(new FileReader(Globals.getTrainFile()), Globals.getBUFFER_SIZE()), Globals.getARFF_BUFFER_SIZE());
			structure = reader.getStructure();
			structure.setClassIndex(structure.numAttributes() - 1);

			Instance current = null;
			while ((current = reader.readInstance(structure)) != null) {

				for (int u = 0; u < n; u++) {
					for (int uval = 0; uval < Globals.getParamsPerAtt()[u]; uval++) {
						if (current.value(u) == uval) {
							w.write("1,");
						} else {
							w.write("0,");
						}
					}
				}

				w.write(current.attribute(n).value((int) current.value(n)) + "");
				w.write("\n");
			} 

			w.close();
			System.out.println("Data Successfully Binarized " + out);

		} else if (Globals.getPreProcessParameter().equalsIgnoreCase("compactDiscretizedFile")) {

			System.out.println("------------------------");
			System.out.println("compactDiscretizedFile");
			System.out.println("Assumption 1: all attributes are already discretized");
			System.out.println("Assumption 2: class is the last attribute");
			System.out.println("------------------------");

			Globals.setSOURCEFILE(sourceFileTrain);
			Instances structure = SUtils.setStructure();

			int N = (int) Globals.getNumberInstances();
			int nc = Globals.getNumClasses();
			int n = Globals.getNumAttributes();

			System.out.println("<num data points, num classes> = <" + N + ", " + n + ", " + nc + ">");

			/* Re-write header */
			String outputFileName = Globals.getTempDirectory() + "/" + Globals.getDataSetName() + "_compactDiscretizedFile.arff"; 
			File out = new File(outputFileName);

			Writer w = new BufferedWriter(new FileWriter(out));

			String header = "";
			header += "@relation '" + outputFileName + "'\n\n";
			w.write(header);

			for (int u = 0; u < n; u++) {
				w.write("@attribute att_" + structure.attribute(u).name()  + "_" + u + " { ");
				for (int uval = 0; uval < Globals.getParamsPerAtt()[u]; uval++) {
					if (uval == Globals.getParamsPerAtt()[u] - 1) {
						w.write(uval + "");
					} else {
						w.write(uval + ", ");
					}
				}
				w.write(" }\n");
			}

			w.write("@attribute class { ");
			for (int v = 0; v < structure.attribute(n).numValues(); v++) {
				if (v == structure.attribute(n).numValues() - 1)   {
					w.write(structure.attribute(n).value(v));
				} else {
					w.write(structure.attribute(n).value(v) + ", ");
				}
			}

			w.write(" }\n");

			w.write("\n@data\n\n");

			ArffReader reader = new ArffReader(new BufferedReader(new FileReader(Globals.getTrainFile()), Globals.getBUFFER_SIZE()), Globals.getARFF_BUFFER_SIZE());
			structure = reader.getStructure();
			structure.setClassIndex(structure.numAttributes() - 1);

			Instance current = null;
			while ((current = reader.readInstance(structure)) != null) {

				for (int u = 0; u < n; u++) {
					w.write(current.value(u) + ",");
				}

				w.write(current.attribute(n).value((int) current.value(n)) + "");
				w.write("\n");
			} 

			w.close();
			System.out.println("Data Successfully compactDiscretizedFile " + out);

		} else if (Globals.getPreProcessParameter().equalsIgnoreCase("BinarizeClass")) {

			System.out.println("------------------------");
			System.out.println("BinarizeClass");
			System.out.println("------------------------");

			Globals.setSOURCEFILE(sourceFileTrain);
			Instances structure = SUtils.setStructure();

			int N = (int) Globals.getNumberInstances();
			int nc = Globals.getNumClasses();
			int n = Globals.getNumAttributes();

			System.out.println("<num data points, num classes> = <" + N + ", " + n + ", " + nc + ">");

			/* Re-write header */
			String outputFileName = Globals.getTempDirectory() + "/" + Globals.getDataSetName() + "_BinarizedClass.arff"; 
			File out = new File(outputFileName);

			Writer w = new BufferedWriter(new FileWriter(out));

			String header = "";
			header += "@relation '" + outputFileName + "'\n\n";
			w.write(header);

			for (int u = 0; u < n; u++) {
				if (structure.attribute(u).isNumeric()) {
					/* numeric attribute */	
					w.write("@attribute att_" + structure.attribute(u).name() + " real \n");
				} else {
					/* Nominal attribute */
					w.write("@attribute dX" + structure.attribute(u).name() + " { ");
					for (int uval = 0; uval < structure.attribute(u).numValues(); uval++) {
						if (uval ==  structure.attribute(u).numValues() - 1) {
							w.write(structure.attribute(u).value(uval));
						} else {
							w.write(structure.attribute(u).value(uval) + ", ");
						}
					}
					w.write(" }\n");
				}
			}

			w.write("@attribute class {0, 1}");

			w.write("\n@data\n\n");

			ArffReader reader = new ArffReader(new BufferedReader(new FileReader(Globals.getTrainFile()), Globals.getBUFFER_SIZE()), Globals.getARFF_BUFFER_SIZE());
			structure = reader.getStructure();
			structure.setClassIndex(structure.numAttributes() - 1);

			int classval = Globals.getClassAttribute();

			Instance current = null;
			while ((current = reader.readInstance(structure)) != null) {

				for (int u = 0; u < n; u++) {
					w.write(current.attribute(u).value((int)current.value(u))+ ",");
				}

				if (current.classValue() == classval) {
					w.write("1");
				} else {
					w.write("0");
				}
				w.write("\n");
			} 

			w.close();
			System.out.println("Data Successfully BinarizedClass " + out);
		}

	}

	private static File collateDiscretizedFiles(File sourceFileTrain, String outputFileName, File[] discreteFiles) throws FileNotFoundException, IOException {

		int classAttribute = Globals.getClassAttribute();

		ArffReader[] readers = new ArffReader[discreteFiles.length];
		Instances[] structures = new Instances[discreteFiles.length];
		for (int u = 0; u < discreteFiles.length; u++) {
			if (discreteFiles[u] != null) {
				readers[u] = new ArffReader(new BufferedReader(new FileReader(discreteFiles[u]), Globals.getBUFFER_SIZE()), Globals.getARFF_BUFFER_SIZE());
				structures[u] = readers[u].getStructure(); 
			}		
		}

		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(Globals.getTrainFile()), Globals.getBUFFER_SIZE()), Globals.getARFF_BUFFER_SIZE());
		Instances structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);

		int n = structure.numAttributes();

		File out = new File(outputFileName);

		Writer w = new BufferedWriter(new FileWriter(out));

		String header = "";
		header += "@relation '" + outputFileName + "'\n\n";
		w.write(header);
		for (int u = 0; u < n; u++) {
			if (u != classAttribute) {
				if (structures[u] == null) {
					/* Already discrete, just copy from the orignal structure */
					w.write("@attribute " + structure.attribute(u).name() + " { ");
					for (int v = 0; v < structure.attribute(u).numValues(); v++) {
						if (v == structure.attribute(u).numValues() - 1) {  
							w.write(structure.attribute(u).value(v));
						} else {
							w.write(structure.attribute(u).value(v) + ", ");
						}
					}
					w.write(" }\n");
				} else {
					w.write("@attribute " + structures[u].attribute(0).name() + " { ");
					for (int v = 0; v < structures[u].attribute(0).numValues(); v++) {
						if (v == structures[u].attribute(0).numValues() - 1) {  
							w.write(structures[u].attribute(0).value(v));
						} else {
							w.write(structures[u].attribute(0).value(v) + ", ");
						}
					}
					w.write(" }\n");			
				}
			}
		}

		w.write("@attribute class { ");
		if (classAttribute == -1) {
			w.write("0 , 1");
		} else {
			for (int v = 0; v < structure.attribute(classAttribute).numValues(); v++) {
				if (v == structure.attribute(classAttribute).numValues() - 1)   {
					w.write(structure.attribute(classAttribute).value(v));
				} else {
					w.write(structure.attribute(classAttribute).value(v) + ", ");
				}
			}
		}
		w.write(" }\n");

		w.write("\n@data\n\n");

		Instance current = null;
		while ((current = reader.readInstance(structure)) != null) {

			for (int u = 0; u < n; u++) {
				if (u != classAttribute) {
					if (structures[u] == null) {
						/* Already discrete, just copy from the original structure */
						if (!current.isMissing(u)) {
							w.write(current.attribute(u).value((int) current.value(u))+ ", ");
						} else {
							w.write("?" + ",");
						}
					} else {
						Instance lcurrent = readers[u].readInstance(structures[u]);

						if (!lcurrent.isMissing(0)) {
							w.write(lcurrent.attribute(0).value((int) lcurrent.value(0)) + ", ");
						} else {
							w.write("?" + ",");
						}

					}
				}
			}

			if (classAttribute == -1) {
				w.write("?");
			} else {
				if (!current.isMissing(classAttribute)) {
					w.write(current.attribute(classAttribute).value((int) current.value(classAttribute)) + "");
				} else {
					w.write("?");
				}
			}

			w.write("\n");
		}

		w.close();
		System.out.println("Data Successfully Collated (discretized OOC) " + out);

		for (int u = 0; u < discreteFiles.length; u++) {
			if (discreteFiles[u] != null) {
				discreteFiles[u].delete(); 
			}		
		}

		return out;
	}

	/*
	 * The following function discretized the in-memory data
	 */
	public static File discretizeDatSet(File data, String outputFileName) throws Exception {

		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(data), Globals.getBUFFER_SIZE()), Globals.getARFF_BUFFER_SIZE());
		Instances structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);

		Instances inMemInstances = new Instances(structure);

		reader = new ArffReader(new BufferedReader(new FileReader(data), Globals.getBUFFER_SIZE()), Globals.getARFF_BUFFER_SIZE());
		Instance current = null;

		while ((current = reader.readInstance(structure)) != null) {
			inMemInstances.add(current);
		}

		/* Discretize the data */
		System.out.println("Starting MDL Discretization");

		weka.filters.supervised.attribute.Discretize m_Disc = null;

		m_Disc = new weka.filters.supervised.attribute.Discretize();
		m_Disc.setUseBinNumbers(true);
		m_Disc.setInputFormat(structure);
		Instances m_DiscreteInstances = weka.filters.Filter.useFilter(inMemInstances, m_Disc);

		File out = new File(outputFileName);

		ArffSaver fileSaver = new ArffSaver();
		fileSaver.setFile(out);
		fileSaver.setStructure(m_DiscreteInstances);
		fileSaver.writeBatch();

		System.out.println("Discretization done successfully");

		/* Delete in memory data */
		inMemInstances = null;
		System.gc();

		return out;
	}

	/* 
	 * The following function slice the data based on inputArguments
	 */

	public static File sliceDataSet(File sourceFileTrain, int[] ignoreAttributes, String outputFileName) throws FileNotFoundException, IOException {

		System.out.println("Attributes to be ignored: " + Arrays.toString(ignoreAttributes));

		int classAttribute = Globals.getClassAttribute();

		if (classAttribute == -1) {
			System.out.println("No Class attribute is provided, will add an extra attribute at the end with ? and header {0, 1} -- Please modify the header yourself");
		} else {
			System.out.println("Class Attribute is at index: " + classAttribute);
		}

		Globals.setSOURCEFILE(sourceFileTrain);

		Instances structure = SUtils.setStructure();

		int N = (int) Globals.getNumberInstances();
		int nc = Globals.getNumClasses();
		int n = Globals.getNumAttributes();

		System.out.println("<num data points, num classes> = <" + N + ", " + n + ", " + nc + ">");

		File out = new File(outputFileName);

		Writer w = new BufferedWriter(new FileWriter(out));

		String header = "";
		header += "@relation '" + outputFileName + "'\n\n";
		w.write(header);

		for (int u = 0; u < n; u++) {
			if (!SUtils.inArray(u, ignoreAttributes)  && u != classAttribute) {
				//if (structure.attribute(u).numValues() > 1) {
				if (structure.attribute(u).isNominal()) {
					/* Discrete Attribute */
					w.write("@attribute " + structure.attribute(u).name() + " { ");

					for (int v = 0; v < structure.attribute(u).numValues(); v++) {
						if (v == structure.attribute(u).numValues() - 1) {  
							w.write(structure.attribute(u).value(v));
						} else {
							w.write(structure.attribute(u).value(v) + ", ");
						}
					}
					w.write(" }\n");
				} else {
					/* Numeric Attribute */
					w.write("@attribute " + structure.attribute(u).name() + " numeric \n");
				}
				//}
			}
		}

		w.write("@attribute class { ");
		if (classAttribute == -1) {
			w.write("0 , 1");
		} else {
			for (int v = 0; v < structure.attribute(classAttribute).numValues(); v++) {
				if (v == structure.attribute(classAttribute).numValues() - 1)   {
					w.write(structure.attribute(classAttribute).value(v));
				} else {
					w.write(structure.attribute(classAttribute).value(v) + ", ");
				}
			}
		}
		w.write(" }\n");

		w.write("\n@data\n\n");

		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFileTrain), Globals.getBUFFER_SIZE()), Globals.getARFF_BUFFER_SIZE());
		Instance current = null;

		while ((current = reader.readInstance(structure)) != null) {

			for (int u = 0; u < n; u++) {
				if (!SUtils.inArray(u, ignoreAttributes) && u != classAttribute) {
					if (!current.isMissing(u)) {
						if (structure.attribute(u).isNominal()) {
							w.write(current.attribute(u).value((int) current.value(u)) + ", ");
						} else {
							w.write(current.value(u) + ",");
						}
					} else {
						w.write("?" +  ",");
					}
				}
			}

			if (classAttribute == -1) {
				w.write("?");
			} else {
				if (!current.isMissing(classAttribute)) {
					w.write(current.attribute(classAttribute).value((int) current.value(classAttribute)) + "");
				} else {
					w.write("?");
				}
			}

			w.write("\n");
		}

		w.close();
		System.out.println("Data Successfully sliced " + out);

		return out;
	}

}
