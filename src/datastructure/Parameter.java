package datastructure;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Map;

import weka.core.Instance;

public abstract class Parameter {

	protected long np;

	protected long originalNP;

	protected int nc;
	protected int n;
	protected double N;

	protected int[] paramsPerAtt;
	protected boolean[] isNumericTrue;
	
	protected int level = 1;

	public double getN() {
		return N;
	}
	
	public int getNC() {
		return nc;
	}
	
	public int getn() {
		return n;
	}
	
	public long getNp() {
		return np;
	}
	
	public long getOriginalNP() {
		return originalNP;
	}

	public void setOriginalNP(long originalNP) {
		this.originalNP = originalNP;
	}
	
	
	
	public int[] getParamsPerAtt() {
		return paramsPerAtt;
	}
	
	public void setNp(long newNP) {
		np = newNP;
	}
	
	public long getTotalNumberParameters() {
		return np;
	}
	
	/*
	 *  -----------------------------------------------
	 * For All
	 *  -----------------------------------------------
	 */

	public abstract void initializeParametersWithVal(int val);

	public abstract void convertToProbs();

	public abstract void updateFirstPass(Instance row);

	public abstract void finishedFirstPass();

	public abstract boolean needSecondPass();

	public abstract void updateAfterFirstPass(Instance row);
	
	
	/*
	 *  -----------------------------------------------
	 * feALR parameter functions
	 *  -----------------------------------------------
	 */
	
	public long determineNP() {
		// Implement in ALR parameter for feALR
		return 0;
	}
	
	/*
	 *  -----------------------------------------------
	 * ALR parameter functions
	 *  -----------------------------------------------
	 */
	
	public void unUpdateAfterFirstPass(Instance row) {
		
	}
	
	public void startFSPass(Map<String, Double> fsScore) throws FileNotFoundException, IOException {
		// Implement in ALR parameter
	}

	public long getAttributeIndex(int att1, int att1valindex, int c) {
		// Implement in ALR parameter
		return 0;
	}

	public long getAttributeIndex(int att1, int att1valindex, int att2, int att2valindex, int c) {
		// Implement in ALR parameter
		return 0;
	}

	public long getAttributeIndex(int att1, int att1valindex, int att2, int att2valindex, int att3, int att3valindex, int c) {
		// Implement in ALR parameter
		return 0;
	}

	public long getAttributeIndex(int att1, int att1valindex, int att2, int att2valindex, int att3, int att3valindex, int att4, int att4valindex, int c) {
		// Implement in ALR parameter
		return 0;
	}

	public long getAttributeIndex(int att1, int att1valindex, int att2, int att2valindex, int att3, int att3valindex, int att4, int att4valindex, int att5, int att5valindex, int c) {
		// Implement in ALR parameter
		return 0;
	}

	public int getCompactIndexAtFullIndex(long index) {
		// Implement in ALR parameter
		return 0;
	}

	public double getProbAtFullIndex(long index) {
		// Implement in ALR parameter
		return 0;
	}

	public double getParameterAtFullIndex(long index) {
		// Implement in ALR parameter
		return 0;
	}

	public void setParameterAtFullIndex(long index, double p) {
		// Implement in ALR parameter
	}
	
	public double getGradientAtFullIndex(long index) {
		// Implement in ALR parameter
		return 0;
	}

	public void setGradientAtFullIndex(long index, double p) {
		// Implement in ALR parameter
	}
	
	/*
	 *  -----------------------------------------------
	 * BNC parameter functions
	 *  -----------------------------------------------
	 */

	public void finishedSecondPass() {
		// TODO Auto-generated method stub
		
	}
	
	public void startThirdPass() throws FileNotFoundException, IOException {
		// Implement in BNC parameter
	}


	public double[] getClassProbabilities() {
		// Implement in BNC parameter
		return null;
	}

	public double[] getParameters() {
		// Implement in BNC parameter
		return null;
	}

	public int[] getOrder() {
		// Implement in BNC parameter
		return null;
	}

	public int[][] getParents() {
		// Implement in BNC parameter
		return null;
	}
	
	/*
	 *  -----------------------------------------------
	 * AnDE parameter functions
	 *  -----------------------------------------------
	 */
	
	public double getCountAtFullIndex(long index) {
		// Implement in AnDE parameter
		return 0;
	}
	
	/*
	 *  -----------------------------------------------
	 * FM
	 *  -----------------------------------------------
	 */
	
//	public double getLatentParameterAtFullIndex(long index, int k) {
//		return 0;
//	}
//	
//	public void setLatentParameterAtFullIndex(long index, int k , double p) {
//		
//	}
	
	public long getAttributeIndex(int att1, int att1valindex, int c, int k) {
		// Implement in FM parameter
		return 0;
	}
	

}
