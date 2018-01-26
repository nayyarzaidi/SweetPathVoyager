package evaluation;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.BitSet;

import utils.Globals;
import utils.SUtils;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.Saver;
import weka.core.converters.ArffLoader.ArffReader;

public class PreprocessData {

	static File out = null;
	static BitSet res = null;
	static Instances CVInstances = null;

	public static File preProcessData() throws Exception {

		if (Globals.isCvFilePresent()) {
			
			System.out.println("Loading CV File in memory");
			CVInstances = SUtils.getTrainTestInstances(Globals.getCVFILE());
			Globals.setCVInstances(CVInstances);
			System.out.println("CV file load successfully in memory");
		
			out = Globals.getSOURCEFILE();
			
		} else {
			
			System.out.println("Start: Getting Stratified Indices");
			res = SUtils.getStratifiedIndices();
			System.out.println("Finish: Getting Stratified Indices");

			System.out.println("Start: Creating CV Instances in memory");
			CVInstances = SUtils.getTrainTestInstances(res);
			System.out.println("Finish: Creating CV Instances in memory");

			Globals.setCVInstances(CVInstances);

			out = File.createTempFile("trainCV-", ".arff");
			out.deleteOnExit();

			PreprocessData.excludeCVInstances();

		}

		if (!Globals.getDiscretization().equalsIgnoreCase("None")) {

			PreprocessData.discreteNumeric();

		} else if (Globals.isNormalizeNumeric()) {

			PreprocessData.normalizeNumeric();

		} 

		if (Globals.isVerbose()) {
			System.out.println("New Source File after Pre-processing is: " + out.getAbsolutePath());
		}

		return out;

	}

	public static void excludeCVInstances() throws Exception {

		System.out.println("Starting Exclusion of CV Instances");

		ArffSaver fileSaver = new ArffSaver();
		fileSaver.setFile(out);
		fileSaver.setRetrieval(Saver.INCREMENTAL);

		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(Globals.getSOURCEFILE()), Globals.getBUFFER_SIZE()), Globals.getARFF_BUFFER_SIZE());
		Instances structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);

		fileSaver.setStructure(structure);

		Instance row;
		int i = 0;
		int lineNo = 0;
		while ((row = reader.readInstance(structure)) != null)  {
			if (!res.get(lineNo)) {

				fileSaver.writeIncremental(row);
				i++;
			}
			lineNo++;
		}

		fileSaver.writeIncremental(null);

		System.out.println("New Training set size = " + i);
	}

	public static void normalizeNumeric() throws Exception {

		System.out.println("Starting Normalization");

		Instances m_NormalizedInstances = null;

		weka.filters.unsupervised.attribute.Normalize m_Norm = null;

		m_Norm = new weka.filters.unsupervised.attribute.Normalize();
		m_Norm.setInputFormat(Globals.getCVInstances());
		m_NormalizedInstances = weka.filters.Filter.useFilter(Globals.getCVInstances(), m_Norm);

		ArffSaver fileSaver = new ArffSaver();
		fileSaver.setFile(out);
		fileSaver.setRetrieval(Saver.INCREMENTAL);
		fileSaver.setStructure(m_NormalizedInstances);

		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(Globals.getSOURCEFILE()), Globals.getBUFFER_SIZE()), Globals.getARFF_BUFFER_SIZE());
		Instances structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);

		Instance row;
		int i = 0;
		int lineNo = 0;
		while ((row = reader.readInstance(structure)) != null)  {
			if (!res.get(lineNo)) {
				m_Norm.input(row);
				row = m_Norm.output();

				fileSaver.writeIncremental(row);
				i++;
			}
			lineNo++;
		}

		fileSaver.writeIncremental(null);

		System.out.println("New Training size = " + i);

		Globals.setCVInstances(m_NormalizedInstances);
	}

	public static void discreteNumeric() throws Exception {

		String val = Globals.getDiscretization();
		int s = Globals.getDiscretizationParameter();

		if (val.equalsIgnoreCase("mdl")) {
			discretizeMDL();

		} if (val.equalsIgnoreCase("ef")) {
			discretizeEF(s);

		} if (val.equalsIgnoreCase("ew")) {
			discretizeEW(s);

		} 

	}

	private static void discretizeEW(int s) {
		// TODO Auto-generated method stub

	}

	private static void discretizeEF( int s) throws Exception {
		/* Equal Frequency based discretization */

		int numBins = s;

		System.out.println("Starting Equal Frequency Discretization with " + numBins + " bins.");

		weka.filters.unsupervised.attribute.Discretize m_Disc = null;

		m_Disc = new weka.filters.unsupervised.attribute.Discretize();
		m_Disc.setUseBinNumbers(true);
		m_Disc.setInputFormat(Globals.getCVInstances());
		m_Disc.setBins(numBins);
		Instances m_DiscreteInstances = weka.filters.Filter.useFilter(Globals.getCVInstances(), m_Disc);

		ArffSaver fileSaver = new ArffSaver();
		fileSaver.setFile(out);
		fileSaver.setRetrieval(Saver.INCREMENTAL);
		fileSaver.setStructure(m_DiscreteInstances);

		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(Globals.getSOURCEFILE()), Globals.getBUFFER_SIZE()), Globals.getARFF_BUFFER_SIZE());
		Instances structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);

		Instance row;
		int i = 0;
		int lineNo = 0;
		while ((row = reader.readInstance(structure)) != null)  {
			if (!res.get(lineNo)) {
				m_Disc.input(row);
				row = m_Disc.output();

				fileSaver.writeIncremental(row);
				i++;
			}
			lineNo++;
		}

		fileSaver.writeIncremental(null);

		System.out.println("New Training size = " + i);

		Globals.setCVInstances(m_DiscreteInstances);
	}

	private static void discretizeMDL() throws Exception {
		/* MDL based discretization */

		System.out.println("Starting MDL Discretization");

		weka.filters.supervised.attribute.Discretize m_Disc = null;

		m_Disc = new weka.filters.supervised.attribute.Discretize();
		m_Disc.setUseBinNumbers(true);
		m_Disc.setInputFormat(Globals.getCVInstances());
		Instances m_DiscreteInstances = weka.filters.Filter.useFilter(Globals.getCVInstances(), m_Disc);

		ArffSaver fileSaver = new ArffSaver();
		fileSaver.setFile(out);
		fileSaver.setRetrieval(Saver.INCREMENTAL);
		fileSaver.setStructure(m_DiscreteInstances);

		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(Globals.getSOURCEFILE()), Globals.getBUFFER_SIZE()), Globals.getARFF_BUFFER_SIZE());
		Instances structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);

		Instance row;
		int i = 0;
		int lineNo = 0;
		while ((row = reader.readInstance(structure)) != null)  {
			if (!res.get(lineNo)) {
				m_Disc.input(row);
				row = m_Disc.output();

				fileSaver.writeIncremental(row);
				i++;

			}
			lineNo++;
		}

		fileSaver.writeIncremental(null);
		System.out.println("New Training size = " + i);

		Globals.setCVInstances(m_DiscreteInstances);
	}

}
