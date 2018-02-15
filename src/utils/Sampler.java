package utils;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;

import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomDataGenerator;
import org.apache.commons.math3.random.RandomGenerator;

import moa.core.InstanceExample;
import moa.streams.ArffFileStream;

import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.InstancesHeader;

import utils.GradualDriftGeneratorLR;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;
import moa.tasks.WriteStreamToARFFFile;

public class Sampler {

	public static File generateNoDrift(int exp, double magnitude) throws IOException {

		/* Bayesian */
		BNGradualDriftMixtureGenerator stream = new BNGradualDriftMixtureGenerator();

		stream.seed.setValue(3071980 + exp);

		stream.nAttributes.setValue(Globals.getDriftNAttributes());
		stream.nValuesPerAttribute.setValue(Globals.getDriftNAttributesValues());

		stream.burnInNInstances.setValue(Globals.getTotalNInstancesBeforeDrift());
		stream.driftLength.setValue(Globals.getTotalNInstancesDuringDrift());

		if (magnitude == 0.0) {
			stream.driftConditional.setValue(false);
		} else {
			stream.driftConditional.setValue(true);
			stream.driftMagnitudeConditional.setValue(magnitude);
		}

		stream.driftPriors.setValue(false);
		stream.precisionDriftMagnitude.setValue(0.05);

		stream.prepareForUse();

		System.out.println("Trying to write file at: " + Globals.getTempDirectory());
		File out = File.createTempFile("trainCV-", ".arff", new File(Globals.getTempDirectory()));
		System.out.println("Creating File at: " +  out.getAbsolutePath());
		out.deleteOnExit();

		Writer w = new BufferedWriter(new FileWriter(out));

		//w.write(stream.getHeader().toString());
		//w.write("\n")

		String header = "";
		header += "@relation '" + stream.getHeader().getRelationName() + "'\n\n";
		for (int i = 0; i < stream.getHeader().numAttributes(); i++) {
			header += "@attribute x" + i + " { ";
			for (int j = 0; j < stream.getHeader().attribute(i).numValues(); j++) {
				if (j == stream.getHeader().attribute(i).numValues() - 1) {
					//header += stream.getHeader().attribute(i).value(j) + ", ";
					header += (double) j;
				} else {
					header += (double) j + ", ";
				}
			}
			header += " }\n";
		}
		header += "\n@data\n\n";

		w.write(header);

		int numWritten = 0;
		int numData = Globals.getTotalNInstancesBeforeDrift() + Globals.getTotalNInstancesDuringDrift() + Globals.getTotalNInstancesAfterDrift();
		while ((numWritten < numData) && stream.hasMoreInstances()) {
			//w.write(stream.nextInstance().toString());
			w.write(stream.nextInstance().instance.toString());
			w.write("\n");
			numWritten++;
		}
		w.close();

		System.out.println("Stream written to ARFF file " + out);

		//		WriteStreamToARFFFile fileSaver = new WriteStreamToARFFFile();
		//		fileSaver.streamOption.setCurrentObject(stream);
		//		fileSaver.maxInstancesOption.setValue(2000000);
		//		fileSaver.arffFileOption.setValue(out.getAbsolutePath());
		//		fileSaver.doTask();

		return out;
	}

	public static File generateDriftGradualBayesian(int exp, double magnitude) throws IOException {

		/* Bayesian */

		BNGradualDriftMixtureGenerator stream = new BNGradualDriftMixtureGenerator();

		stream.seed.setValue(3071980 + exp);

		stream.nAttributes.setValue(Globals.getDriftNAttributes());
		stream.nValuesPerAttribute.setValue(Globals.getDriftNAttributesValues());

		stream.burnInNInstances.setValue(Globals.getTotalNInstancesBeforeDrift());
		stream.driftLength.setValue(Globals.getTotalNInstancesDuringDrift());

		if (magnitude == 0.0) {
			stream.driftConditional.setValue(false);
		} else {
			stream.driftConditional.setValue(true);
			stream.driftMagnitudeConditional.setValue(magnitude);
		}

		stream.driftPriors.setValue(false);
		stream.precisionDriftMagnitude.setValue(0.05);
		stream.prepareForUse();

		System.out.println("Trying to write file at: " + Globals.getTempDirectory());
		File out = File.createTempFile("trainCV-", ".arff", new File(Globals.getTempDirectory()));
		System.out.println("Creating File at: " +  out.getAbsolutePath());
		out.deleteOnExit();

		Writer w = new BufferedWriter(new FileWriter(out));

		//w.write(stream.getHeader().toString());
		//w.write("\n")

		String header = "";
		header += "@relation '" + stream.getHeader().getRelationName() + "'\n\n";
		for (int i = 0; i < stream.getHeader().numAttributes(); i++) {
			header += "@attribute x" + i + " { ";
			for (int j = 0; j < stream.getHeader().attribute(i).numValues(); j++) {
				if (j == stream.getHeader().attribute(i).numValues() - 1) {
					//header += stream.getHeader().attribute(i).value(j) + ", ";
					header += (double) j;
				} else {
					header += (double) j + ", ";
				}
			}
			header += " }\n";
		}
		header += "\n@data\n\n";

		w.write(header);

		int numWritten = 0;
		int numData = Globals.getTotalNInstancesBeforeDrift() + Globals.getTotalNInstancesDuringDrift() + Globals.getTotalNInstancesAfterDrift();
		while ((numWritten < numData) && stream.hasMoreInstances()) {
			//w.write(stream.nextInstance().toString());
			w.write(stream.nextInstance().instance.toString());
			w.write("\n");
			numWritten++;
		}
		w.close();

		System.out.println("Stream written to ARFF file " + out);

		//		WriteStreamToARFFFile fileSaver = new WriteStreamToARFFFile();
		//		fileSaver.streamOption.setCurrentObject(stream);
		//		fileSaver.maxInstancesOption.setValue(2000000);
		//		fileSaver.arffFileOption.setValue(out.getAbsolutePath());
		//		fileSaver.doTask();

		return out;
	}

	public static File generateDriftGradual(int exp, double magnitude) throws IOException {

		/* LR */

		GradualDriftGeneratorLR stream = new GradualDriftGeneratorLR();

		stream.nAttributes.setValue(Globals.getDriftNAttributes());
		stream.nValuesPerAttribute.setValue(Globals.getDriftNAttributesValues());
		stream.seed.setValue(3071980 + exp);
		stream.burnInNInstances.setValue(Globals.getTotalNInstancesBeforeDrift());
		if (magnitude == 0.0) {
			stream.driftConditional.setValue(false);
		} else {
			stream.driftConditional.setValue(true);
			stream.driftMagnitudeConditional.setValue(magnitude);
		}
		stream.driftPriors.setValue(false);
		stream.driftLength.setValue(Globals.getTotalNInstancesDuringDrift());
		stream.precisionDriftMagnitude.setValue(0.05);
		stream.prepareForUse();

		File out = File.createTempFile("trainCV-", ".arff", new File(Globals.getTempDirectory()));
		System.out.println("Creating File at: " +  out.getAbsolutePath());
		out.deleteOnExit();

		Writer w = new BufferedWriter(new FileWriter(out));

		//w.write(stream.getHeader().toString());
		//w.write("\n")

		String header = "";
		header += "@relation '" + stream.getHeader().getRelationName() + "'\n\n";
		for (int i = 0; i < stream.getHeader().numAttributes(); i++) {
			header += "@attribute x" + i + " { ";
			for (int j = 0; j < stream.getHeader().attribute(i).numValues(); j++) {
				if (j == stream.getHeader().attribute(i).numValues() - 1) {
					//header += stream.getHeader().attribute(i).value(j) + ", ";
					header += (double) j;
				} else {
					header += (double) j + ", ";
				}
			}
			header += " }\n";
		}
		header += "\n@data\n\n";

		w.write(header);

		int numWritten = 0;
		int numData = Globals.getTotalNInstancesBeforeDrift() + Globals.getTotalNInstancesDuringDrift() + Globals.getTotalNInstancesAfterDrift();
		while ((numWritten < numData) && stream.hasMoreInstances()) {
			//w.write(stream.nextInstance().toString());
			w.write(stream.nextInstance().instance.toString());
			w.write("\n");
			numWritten++;
		}
		w.close();

		System.out.println("Stream written to ARFF file " + out);

		//		WriteStreamToARFFFile fileSaver = new WriteStreamToARFFFile();
		//		fileSaver.streamOption.setCurrentObject(stream);
		//		fileSaver.maxInstancesOption.setValue(2000000);
		//		fileSaver.arffFileOption.setValue(out.getAbsolutePath());
		//		fileSaver.doTask();

		return out;
	}

	public static File generateDriftGradualSwappingGenerator(int exp, double magnitude) throws IOException {

		/* Bayesian modified by Francois to incorporate Geoff's idea*/

		BNGradualDriftSwappingGenerator stream = new BNGradualDriftSwappingGenerator();

		stream.nAttributes.setValue(Globals.getDriftNAttributes());
		stream.nValuesPerAttribute.setValue(Globals.getDriftNAttributesValues());		
		stream.seed.setValue(3071980 + exp);
		stream.burnInNInstances.setValue(Globals.getTotalNInstancesBeforeDrift());
		stream.driftLength.setValue(Globals.getTotalNInstancesDuringDrift());
		stream.driftConditional.setValue(true);
		stream.driftMagnitudeConditional.setValue(magnitude);
		stream.prepareForUse();

		System.out.println("Trying to write file at: " + Globals.getTempDirectory());
		File out = File.createTempFile("trainCV-", ".arff", new File(Globals.getTempDirectory()));
		System.out.println("Creating File at: " +  out.getAbsolutePath());
		out.deleteOnExit();

		Writer w = new BufferedWriter(new FileWriter(out));

		//w.write(stream.getHeader().toString());
		//w.write("\n")

		String header = "";
		header += "@relation '" + stream.getHeader().getRelationName() + "'\n\n";
		for (int i = 0; i < stream.getHeader().numAttributes(); i++) {
			header += "@attribute x" + i + " { ";
			for (int j = 0; j < stream.getHeader().attribute(i).numValues(); j++) {
				if (j == stream.getHeader().attribute(i).numValues() - 1) {
					//header += stream.getHeader().attribute(i).value(j) + ", ";
					header += (double) j;
				} else {
					header += (double) j + ", ";
				}
			}
			header += " }\n";
		}
		header += "\n@data\n\n";

		w.write(header);

		int numWritten = 0;
		int numData = Globals.getTotalNInstancesBeforeDrift() + Globals.getTotalNInstancesDuringDrift() + Globals.getTotalNInstancesAfterDrift();
		while ((numWritten < numData) && stream.hasMoreInstances()) {
			//w.write(stream.nextInstance().toString());
			w.write(stream.nextInstance().instance.toString());
			w.write("\n");
			numWritten++;
		}
		w.close();

		System.out.println("Stream written to ARFF file " + out);

		//		WriteStreamToARFFFile fileSaver = new WriteStreamToARFFFile();
		//		fileSaver.streamOption.setCurrentObject(stream);
		//		fileSaver.maxInstancesOption.setValue(2000000);
		//		fileSaver.arffFileOption.setValue(out.getAbsolutePath());
		//		fileSaver.doTask();

		return out;
	}

	public static File generateDriftData(int exp, double magnitude) throws IOException {

		/* ---------------------------------------------------------------------- */
		int totalNInstancesBeforeDrift = Globals.getTotalNInstancesBeforeDrift();
		int totalNInstancesAfterDrift = Globals.getTotalNInstancesAfterDrift();

		int nAttributes = Globals.getDriftNAttributes();
		int nValuesPerAttribute = Globals.getDriftNAttributesValues();

		boolean randomizePrior = true;

		int nCombinationsValuesForPX = 1;
		for (int a = 0; a < nAttributes; a++) {
			nCombinationsValuesForPX *= nValuesPerAttribute;
		}

		double[][] pxbd = new double[nAttributes][nValuesPerAttribute];
		double[][] pygx = new double[nCombinationsValuesForPX][nValuesPerAttribute];

		RandomGenerator rg = new JDKRandomGenerator();
		rg.setSeed(3071980 + exp);

		RandomDataGenerator r = new RandomDataGenerator(rg);

		// generate p(x)
		DriftGenerator.generateRandomPx(pxbd, r);
		/* ---------------------------------------------------------------------- */

		rg = new JDKRandomGenerator();
		rg.setSeed(exp);
		r = new RandomDataGenerator(rg);
		// generate new starting pxbd for this experiment
		DriftGenerator.generateRandomPyGivenX(pygx, r);
		if (randomizePrior) {
			DriftGenerator.generateRandomPx(pxbd, r);
		}

		double[][] pygxad = generatePYGivenxWithMag(pygx, magnitude, r, nCombinationsValuesForPX);

		//AbruptDriftGenerator stream = new AbruptDriftGenerator();
		AbruptDriftGeneratorWithParameters stream = new AbruptDriftGeneratorWithParameters();

		stream.nAttributes.setValue(nAttributes);
		stream.nValuesPerAttribute.setValue(nValuesPerAttribute);
		stream.seed.setValue(exp);
		stream.burnInNInstances.setValue(totalNInstancesBeforeDrift);
		stream.setPrioDistBeforeDrift(pxbd);
		stream.setPrioDistAfterDrift(pxbd);
		stream.setCondDistBeforeDrift(pygx);
		stream.setCondDistAfterDrift(pygxad);
		stream.prepareForUse();

		File out = File.createTempFile("trainCV-", ".arff", new File(Globals.getTempDirectory()));
		System.out.println("Creating File at: " +  out.getAbsolutePath());
		out.deleteOnExit();

		Writer w = new BufferedWriter(new FileWriter(out));

		//w.write(stream.getHeader().toString());
		//w.write("\n")

		String header = "";
		header += "@relation '" + stream.getHeader().getRelationName() + "'\n\n";
		for (int i = 0; i < stream.getHeader().numAttributes(); i++) {
			header += "@attribute x" + i + " { ";
			for (int j = 0; j < stream.getHeader().attribute(i).numValues(); j++) {
				if (j == stream.getHeader().attribute(i).numValues() - 1) {
					//header += stream.getHeader().attribute(i).value(j) + ", ";
					header += (double) j;
				} else {
					header += (double) j + ", ";
				}
			}
			header += " }\n";
		}
		header += "\n@data\n\n";

		w.write(header);

		int numWritten = 0;
		int numData = Globals.getTotalNInstancesBeforeDrift() + Globals.getTotalNInstancesDuringDrift() + Globals.getTotalNInstancesAfterDrift();
		while ((numWritten < numData) && stream.hasMoreInstances()) {
			//w.write(stream.nextInstance().toString());
			w.write(stream.nextInstance().instance.toString());
			w.write("\n");
			numWritten++;
		}
		w.close();

		System.out.println("Stream written to ARFF file " + out);

		//		WriteStreamToARFFFile fileSaver = new WriteStreamToARFFFile();
		//		fileSaver.streamOption.setCurrentObject(stream);
		//		fileSaver.maxInstancesOption.setValue(2000000);
		//		fileSaver.arffFileOption.setValue(out.getAbsolutePath());
		//		fileSaver.doTask();

		return out;
	}

	public static double[][] generatePYGivenxWithMag(double[][] pygx, double mag, RandomDataGenerator r,
			int nCombinationsValuesForPX) {
		int nLinesToChange = (int) Math.round(mag * nCombinationsValuesForPX);
		if (nLinesToChange == 0.0) {
			return pygx;
		}
		double[][] pygxad = new double[nCombinationsValuesForPX][];
		for (int line = 0; line < pygxad.length; line++) {
			// default is same distrib
			pygxad[line] = pygx[line];
		}
		int[] linesToChange = r.nextPermutation(nCombinationsValuesForPX, nLinesToChange);

		for (int line : linesToChange) {
			pygxad[line] = new double[pygx[line].length];

			double[] lineCPT = pygxad[line];
			int chosenClass;

			do {
				chosenClass = r.nextInt(0, lineCPT.length - 1);
				// making sure we choose a different class value
			} while (pygx[line][chosenClass] == 1.0);

			for (int c = 0; c < lineCPT.length; c++) {
				if (c == chosenClass) {
					lineCPT[c] = 1.0;
				} else {
					lineCPT[c] = 0.0;
				}
			}
		}
		return pygxad;
	}
	
	public static File generateTANDrift(int exp, double frequency) throws IOException {

		//SimpleTANGenerator stream = new SimpleTANGenerator();
		SimpleTANGeneratorWithDriftBinaryValuesMod stream = new SimpleTANGeneratorWithDriftBinaryValuesMod();
		
		stream.frequency.setValue((int) frequency);

		stream.seed.setValue(3071980 + exp);

		stream.nAttributes.setValue(Globals.getDriftNAttributes());
		stream.nValuesPerAttribute.setValue(Globals.getDriftNAttributesValues());

		stream.driftLength.setValue(Globals.getTotalNInstancesDuringDrift());

		stream.prepareForUse();

		System.out.println("Trying to write file at: " + Globals.getTempDirectory());
		File out = File.createTempFile("trainCV-", ".arff", new File(Globals.getTempDirectory()));
		System.out.println("Creating File at: " +  out.getAbsolutePath());
		out.deleteOnExit();

		Writer w = new BufferedWriter(new FileWriter(out));

		String header = "";
		header += "@relation '" + stream.getHeader().getRelationName() + "'\n\n";
		for (int i = 0; i < stream.getHeader().numAttributes(); i++) {
			header += "@attribute x" + i + " { ";
			for (int j = 0; j < stream.getHeader().attribute(i).numValues(); j++) {
				if (j == stream.getHeader().attribute(i).numValues() - 1) {
					//header += stream.getHeader().attribute(i).value(j) + ", ";
					header += (double) j;
				} else {
					header += (double) j + ", ";
				}
			}
			header += " }\n";
		}
		header += "\n@data\n\n";

		w.write(header);

		int numWritten = 0;
		int numData = Globals.getTotalNInstancesBeforeDrift() + Globals.getTotalNInstancesDuringDrift() + Globals.getTotalNInstancesAfterDrift();
		while ((numWritten < numData) && stream.hasMoreInstances()) {
			w.write(stream.nextInstance().instance.toString());
			w.write("\n");
			numWritten++;
		}
		w.close();

		System.out.println("Stream written to ARFF file " + out);

		return out;
	}

	public static File generateKDBDrift(int exp, double frequency) throws IOException {

		SimpleKDBGeneratorWithDriftBinaryValues stream = new SimpleKDBGeneratorWithDriftBinaryValues();
		
		stream.frequency.setValue((int) frequency);

		stream.seed.setValue(3071980 + exp);

		stream.nAttributes.setValue(Globals.getDriftNAttributes());
		stream.nValuesPerAttribute.setValue(Globals.getDriftNAttributesValues());

		stream.driftLength.setValue(Globals.getTotalNInstancesDuringDrift());

		stream.prepareForUse();

		System.out.println("Trying to write file at: " + Globals.getTempDirectory());
		File out = File.createTempFile("trainCV-", ".arff", new File(Globals.getTempDirectory()));
		System.out.println("Creating File at: " +  out.getAbsolutePath());
		out.deleteOnExit();

		Writer w = new BufferedWriter(new FileWriter(out));

		String header = "";
		header += "@relation '" + stream.getHeader().getRelationName() + "'\n\n";
		for (int i = 0; i < stream.getHeader().numAttributes(); i++) {
			header += "@attribute x" + i + " { ";
			for (int j = 0; j < stream.getHeader().attribute(i).numValues(); j++) {
				if (j == stream.getHeader().attribute(i).numValues() - 1) {
					//header += stream.getHeader().attribute(i).value(j) + ", ";
					header += (double) j;
				} else {
					header += (double) j + ", ";
				}
			}
			header += " }\n";
		}
		header += "\n@data\n\n";

		w.write(header);

		int numWritten = 0;
		int numData = Globals.getTotalNInstancesBeforeDrift() + Globals.getTotalNInstancesDuringDrift() + Globals.getTotalNInstancesAfterDrift();
		while ((numWritten < numData) && stream.hasMoreInstances()) {
			w.write(stream.nextInstance().instance.toString());
			w.write("\n");
			numWritten++;
		}
		w.close();

		System.out.println("Stream written to ARFF file " + out);

		return out;
	}

	public static File generateSwitchingKDBDrift(int exp, double frequency) throws IOException {

		System.out.println("Generating");
		SwitchingKDBGeneratorWithDriftBinaryValues stream = new SwitchingKDBGeneratorWithDriftBinaryValues();

		stream.frequency.setValue((int) frequency);

		stream.seed.setValue(3071980 + exp);

		stream.nAttributes.setValue(Globals.getDriftNAttributes());
		stream.nValuesPerAttribute.setValue(Globals.getDriftNAttributesValues());

		stream.driftLength.setValue(Globals.getTotalNInstancesDuringDrift());
		stream.dirChangeProb.setValue(0.01);

		stream.prepareForUse();

		System.out.println("Trying to write file at: " + Globals.getTempDirectory());
		File out = File.createTempFile("trainCV-", ".arff", new File(Globals.getTempDirectory()));
		System.out.println("Creating File at: " +  out.getAbsolutePath());
		out.deleteOnExit();

		Writer w = new BufferedWriter(new FileWriter(out));

		String header = "";
		header += "@relation '" + stream.getHeader().getRelationName() + "'\n\n";
		for (int i = 0; i < stream.getHeader().numAttributes(); i++) {
			header += "@attribute x" + i + " { ";
			for (int j = 0; j < stream.getHeader().attribute(i).numValues(); j++) {
				if (j == stream.getHeader().attribute(i).numValues() - 1) {
					//header += stream.getHeader().attribute(i).value(j) + ", ";
					header += (double) j;
				} else {
					header += (double) j + ", ";
				}
			}
			header += " }\n";
		}
		header += "\n@data\n\n";

		w.write(header);

		int numWritten = 0;
		int numData = Globals.getTotalNInstancesBeforeDrift() + Globals.getTotalNInstancesDuringDrift() + Globals.getTotalNInstancesAfterDrift();
		while ((numWritten < numData) && stream.hasMoreInstances()) {
			w.write(stream.nextInstance().instance.toString());
			w.write("\n");
			numWritten++;
		}
		w.close();

		System.out.println("Stream written to ARFF file " + out);

		return out;
	}

	public static File generateSimpleDrift(int exp, double frequency) throws IOException {

		/* Simple (Geoff) */

		SimpleDriftGenerator stream = new SimpleDriftGenerator();
		stream.frequency.setValue((int) frequency);

		stream.seed.setValue(3071980 + exp);

		stream.nAttributes.setValue(Globals.getDriftNAttributes());
		stream.nValuesPerAttribute.setValue(Globals.getDriftNAttributesValues());

		stream.driftLength.setValue(Globals.getTotalNInstancesDuringDrift());

		stream.prepareForUse();

		System.out.println("Trying to write file at: " + Globals.getTempDirectory());
		File out = File.createTempFile("trainCV-", ".arff", new File(Globals.getTempDirectory()));
		System.out.println("Creating File at: " +  out.getAbsolutePath());
		out.deleteOnExit();

		Writer w = new BufferedWriter(new FileWriter(out));

		//w.write(stream.getHeader().toString());
		//w.write("\n")

		String header = "";
		header += "@relation '" + stream.getHeader().getRelationName() + "'\n\n";
		for (int i = 0; i < stream.getHeader().numAttributes(); i++) {
			header += "@attribute x" + i + " { ";
			for (int j = 0; j < stream.getHeader().attribute(i).numValues(); j++) {
				if (j == stream.getHeader().attribute(i).numValues() - 1) {
					//header += stream.getHeader().attribute(i).value(j) + ", ";
					header += (double) j;
				} else {
					header += (double) j + ", ";
				}
			}
			header += " }\n";
		}
		header += "\n@data\n\n";

		w.write(header);

		int numWritten = 0;
		int numData = Globals.getTotalNInstancesBeforeDrift() + Globals.getTotalNInstancesDuringDrift() + Globals.getTotalNInstancesAfterDrift();
		while ((numWritten < numData) && stream.hasMoreInstances()) {
			//w.write(stream.nextInstance().toString());
			w.write(stream.nextInstance().instance.toString());
			w.write("\n");
			numWritten++;
		}
		w.close();

		System.out.println("Stream written to ARFF file " + out);

		//		WriteStreamToARFFFile fileSaver = new WriteStreamToARFFFile();
		//		fileSaver.streamOption.setCurrentObject(stream);
		//		fileSaver.maxInstancesOption.setValue(2000000);
		//		fileSaver.arffFileOption.setValue(out.getAbsolutePath());
		//		fileSaver.doTask();

		return out;
	}

	public static File generateSimpleDriftFromData(int exp, double numCycles) throws IOException {

		// Add randomization here
		File sourceFile = Globals.getSOURCEFILE();
		sourceFile = SUtils.randomizeTrainingFile();

		Instances structure = SUtils.setStructure();
		
		File out = File.createTempFile("trainCV-", ".arff", new File(Globals.getTempDirectory()));
		System.out.println("(generateSimpleDriftFromData) Creating File at: " +  out.getAbsolutePath());
		out.deleteOnExit();

		Writer w = new BufferedWriter(new FileWriter(out));

		String header = "";
		header += "@relation '" + "contrieved" + "'\n\n";
		for (int i = 0; i < structure.numAttributes(); i++) {
			header += "@attribute x" + i + " { ";
			for (int j = 0; j < structure.attribute(i).numValues(); j++) {
				if (j == structure.attribute(i).numValues() - 1) {
					header += structure.attribute(i).value(j);
					//header += (double) j;
				} else {
					header += structure.attribute(i).value(j) + ", ";
					//header += (double) j + ", ";
				}
			}
			header += " }\n";
		}
		header += "\n@data\n\n";

		w.write(header);

		for (int r = 0; r < numCycles; r++) {
			
			ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), Globals.getBUFFER_SIZE()), Globals.getARFF_BUFFER_SIZE());
			structure = reader.getStructure();
			structure.setClassIndex(structure.numAttributes() - 1);

			Instance current = null;
			while ((current = reader.readInstance(structure)) != null) {

				for (int u = 0; u < structure.numAttributes() - 1; u++) {
					w.write(current.attribute(u).value((int)current.value(u))+ ",");
				}
				
				w.write(current.attribute(structure.numAttributes()-1).value((int) current.value(structure.numAttributes()-1)));
				w.write("\n");
			}
			
		}
		w.close();

		System.out.println("Stream written to ARFF file " + out);

		return out;
	}

}
