package model.ande;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;

import datastructure.Parameter;
import datastructure.ande.wdAnDEParametersFlat;
import datastructure.ande.wdAnDEParametersIndexedBig;
import model.Model;
import utils.Globals;
import utils.SUtils;
import weka.core.converters.ArffLoader.ArffReader;
import weka.core.Instance;
import weka.core.Instances;

public class ande extends Model {

	private int m_NumTuples = 0; 	                 					

	private Parameter dParameters_ = null;

	private Instances structure = null;

	private File sourceFile = null;

	@Override
	public void buildClassifier() throws Exception {

		sourceFile = Globals.getSOURCEFILE();

		m_NumTuples = Globals.getLevel();

		if (m_NumTuples < 0 || m_NumTuples > 2) {
			System.out.println("AnDE is not implemented with level" + m_NumTuples);
			System.exit(-1);
		}

		System.out.println("[----- AnDE -----]: Level = " + m_NumTuples + " Reading structure -- " + sourceFile);

		if (Globals.getExperimentType().equalsIgnoreCase("prequential") || 
				Globals.getExperimentType().equalsIgnoreCase("flowMachines") || 
				Globals.getExperimentType().equalsIgnoreCase("drift")) {

			/* Prequential, flowMachines Training of ANDE */

			String val = Globals.getDataStructureParameter();

			if (val.equalsIgnoreCase("Flat")) {

				dParameters_ = new wdAnDEParametersFlat();

			} else { 

				System.out.println("Prequential training of AnDE requires only Flat Parameter Structure");
				System.exit(-1);
			}	


		} else {

			/* Typical Training of ANDE */

			String val = Globals.getDataStructureParameter();

			if (val.equalsIgnoreCase("Flat")) {

				dParameters_ = new wdAnDEParametersFlat();

			} else if (val.equalsIgnoreCase("IndexedBig")) {

				dParameters_ = new wdAnDEParametersIndexedBig();

			} else if (val.equalsIgnoreCase("BitMap")) {

				//dParameters_ = new wdAnDEParametersBitmap();

			} else if (val.equalsIgnoreCase("Hash")) {

				//dParameters_ = new wdAnDEParametersHash();

			}	

			// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
			// ^^^^^^^^^^^^^^^^^^^ Pass 1 and (optional) 2 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
			// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
			ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), Globals.getBUFFER_SIZE()), Globals.getARFF_BUFFER_SIZE());
			this.structure = reader.getStructure();
			structure.setClassIndex(structure.numAttributes() - 1);

			Instance row;
			while ((row = reader.readInstance(structure)) != null) {
				dParameters_.updateFirstPass(row);
			}

			if (Globals.isVerbose())
				System.out.println("Finished first pass.");

			dParameters_.finishedFirstPass();

			if (dParameters_.needSecondPass() ) {

				reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), Globals.getBUFFER_SIZE()), Globals.getARFF_BUFFER_SIZE());
				this.structure = reader.getStructure();
				structure.setClassIndex(structure.numAttributes() - 1);

				while ((row = reader.readInstance(structure)) != null) {
					dParameters_.updateAfterFirstPass(row);
				}

				if (Globals.isVerbose())
					System.out.println("Finished second pass.");
			}

			System.out.println("Finish training");
		}

	}

	@Override
	public void update(Instance row) {

		dParameters_.updateFirstPass(row);

	}

	private double[] predictA0DE(Instance inst) {

		int nc = dParameters_.getNC();
		double N = dParameters_.getN();
		int n = dParameters_.getn();
		int[] paramsPerAtt = dParameters_.getParamsPerAtt(); 

		double[] probs = new double[nc];

		for (int c = 0; c < probs.length; c++) {
			//System.out.println(dParameters_.getCountAtFullIndex(c) + ", " + N + ", " + nc);
			probs[c] = Math.log(SUtils.MEsti(dParameters_.getCountAtFullIndex(c), N, nc));

			for (int att1 = 0; att1 < n; att1++) {
				int att1val = (int) inst.value(att1);

				long index = dParameters_.getAttributeIndex(att1, att1val, c);				

				//System.out.println(dParameters_.getCountAtFullIndex((int)index) + ", " + dParameters_.getCountAtFullIndex(c) + ", " + paramsPerAtt[att1]);

				probs[c] += Math.log(SUtils.MEsti(dParameters_.getCountAtFullIndex((int)index), 
						dParameters_.getCountAtFullIndex(c), paramsPerAtt[att1]));
			}
		}

		return probs;	
	}

	private double[] predictA1DE(Instance inst) {

		int nc = dParameters_.getNC();
		double N = dParameters_.getN();
		int n = dParameters_.getn();
		int[] paramsPerAtt = dParameters_.getParamsPerAtt();

		double[] probs = new double[nc];

		double probInitializerA1DE = Double.MAX_VALUE/(n+1);
		double[][] spodeProbs = new double[n][nc];
		int parentCount = 0;

		for (int up = 0; up < n; up++) {
			int x_up = (int) inst.value(up);

			long index = 0;
			int countOfX1AndY = 0;
			for (int c = 0; c < nc; c++) {
				index = dParameters_.getAttributeIndex(up, x_up, c);
				countOfX1AndY += dParameters_.getCountAtFullIndex((int)index);
			}

			// Check that attribute value has a frequency of m_Limit or greater
			if (countOfX1AndY > SUtils.m_Limit) {
				parentCount++;

				for (int c = 0; c < nc; c++) {
					index = dParameters_.getAttributeIndex(up, x_up, c);

					spodeProbs[up][c] = probInitializerA1DE * SUtils.MEsti(dParameters_.getCountAtFullIndex((int)index), N, paramsPerAtt[up] * nc);
				}
			}							
		}

		//		// Check that atleast one parent is used, otherwise, do naive Bayes.
		//		if (parentCount < 1) {
		//			//System.out.println("Resorting to NB");			
		//			return predictA0DE(inst);
		//		} else {

		for (int up = 1; up < n; up++) { // Parent
			int x_up = (int) inst.value(up);

			for (int uc = 0; uc < up; uc++) { // Child				
				int x_uc = (int) inst.value(uc);

				for (int c = 0; c < nc; c++) {
					long index1 = dParameters_.getAttributeIndex(up, x_up, uc, x_uc, c);
					long index2 = dParameters_.getAttributeIndex(uc, x_uc, c);
					long index3 = dParameters_.getAttributeIndex(up, x_up, c);

					spodeProbs[uc][c] *= SUtils.MEsti(dParameters_.getCountAtFullIndex((int)index1), dParameters_.getCountAtFullIndex((int)index2), paramsPerAtt[up]);
					spodeProbs[up][c] *= SUtils.MEsti(dParameters_.getCountAtFullIndex((int)index1), dParameters_.getCountAtFullIndex((int)index3), paramsPerAtt[uc]);
				}					
			}
		}

		/* add all the probabilities for each class */
		for (int c = 0; c < nc; c++) {
			for (int u = 0; u < n; u++) {
				probs[c] += spodeProbs[u][c];
			}			
		}

		SUtils.log(probs);
		//}

		return probs;	

	}

	private double[] predictA2DE(Instance inst) {

		int nc = dParameters_.getNC();
		double N = dParameters_.getN();
		int n = dParameters_.getn();
		int[] paramsPerAtt = dParameters_.getParamsPerAtt();

		double[] probs = new double[nc];

		double probInitializerA2DE = Double.MAX_VALUE/((n+1)*(n+1));
		double[][][] spodeProbs = new double[n][][];
		int parentCount = 0;

		for (int up1 = 1; up1 < n; up1++) {
			int up2size = 0;
			for (int up2 = 0; up2 < up1; up2++) {
				up2size++;
			}
			spodeProbs[up1]  = new double[up2size][nc];
		}

		for (int up1 = 1; up1 < n; up1++) {
			int x_up1 = (int) inst.value(up1);

			for (int up2 = 0; up2 < up1; up2++) {
				int x_up2 = (int) inst.value(up2);

				long index = 0;
				int countOfX1AndX2AndY = 0;
				for (int c = 0; c < nc; c++) {
					index = dParameters_.getAttributeIndex(up1, x_up1, up2, x_up2, c);
					countOfX1AndX2AndY += dParameters_.getCountAtFullIndex((int)index);
				}

				// Check that attribute value has a frequency of m_Limit or greater
				if (countOfX1AndX2AndY >= SUtils.m_Limit) {
					parentCount++;

					for (int c = 0; c < nc; c++) {
						index = dParameters_.getAttributeIndex(up1, x_up1, up2, x_up2, c);
						spodeProbs[up1][up2][c] = probInitializerA2DE * SUtils.MEsti(dParameters_.getCountAtFullIndex((int)index), N, paramsPerAtt[up1] * paramsPerAtt[up2] * nc);						
					}
				}							
			}
		}

		//		// Check that atleast one parent is used, otherwise, do A1DE.
		//		if (parentCount < 1) {
		//			//System.out.println("Resorting to A1DE");
		//			return predictA1DE(inst);
		//		} else {

		for (int up1 = 2; up1 < n; up1++) { // Parent1
			int x_up1 = (int) inst.value(up1);

			for (int up2 = 1; up2 < up1; up2++) { // Parent2
				int x_up2 = (int) inst.value(up2);

				for (int uc = 0; uc < up2; uc++) { // Child
					int x_uc = (int) inst.value(uc);

					for (int c = 0; c < nc; c++) {	// Class

						long index2 = dParameters_.getAttributeIndex(up1, x_up1, up2, x_up2, c);

						long index = dParameters_.getAttributeIndex(up1, x_up1, up2, x_up2, uc, x_uc, c);
						double parentFreq = dParameters_.getCountAtFullIndex((int)index);

						long index3 = dParameters_.getAttributeIndex(up2, x_up2, uc, x_uc, c);
						long index4 = dParameters_.getAttributeIndex(up1, x_up1, uc, x_uc, c);

						spodeProbs[up1][up2][c] *= SUtils.MEsti(parentFreq, dParameters_.getCountAtFullIndex((int)index2), paramsPerAtt[uc]);
						spodeProbs[up2][uc][c] *= SUtils.MEsti(parentFreq, dParameters_.getCountAtFullIndex((int)index3), paramsPerAtt[up1]);
						spodeProbs[up1][uc][c] *= SUtils.MEsti(parentFreq, dParameters_.getCountAtFullIndex((int)index4), paramsPerAtt[up2]);
					}					
				}
			}
		}

		/* add all the probabilities for each class */
		for (int c = 0; c < nc; c++) {
			for (int up1 = 1; up1 < n; up1++) {
				for (int up2 = 0; up2 < up1; up2++) {
					probs[c] += spodeProbs[up1][up2][c];
				}
			}			
		}

		SUtils.log(probs);
		//}

		return probs;	

	}

	@Override
	public double[] distributionForInstance(Instance inst) {
		double[] probs = null;

		if (m_NumTuples == 0) {	
			probs = predictA0DE(inst);
		} else if (m_NumTuples == 1) {
			probs = predictA1DE(inst);
		} else if (m_NumTuples == 2) {
			probs = predictA2DE(inst);
		}

		SUtils.normalizeInLogDomain(probs);
		SUtils.exp(probs);
		return probs;
	}


	@Override
	public double evaluateFunction(File sourceFile) throws IOException {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double[] predict(Instance inst) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void computeGrad(Instance inst, double[] probs, int x_C) {
		// TODO Auto-generated method stub

	}

	@Override
	public void computeGradAndUpdateParameters(Instance instance, double[] probs, int x_C) {
		// TODO Auto-generated method stub

	}

	@Override
	public double[] evaluateFunction(Instances cvInstances) {
		// TODO Auto-generated method stub
		return null;
	}


}
