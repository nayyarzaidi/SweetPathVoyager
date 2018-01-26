package datastructure.ande;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.time.Instant;

import datastructure.Parameter;
import datastructure.indexTrie;
import utils.Globals;
import weka.core.Instance;

public abstract class wdAnDEParameters extends Parameter {

	protected static int MAX_TAB_LENGTH = Integer.MAX_VALUE-8;

	protected int scheme;

	protected indexTrie[] indexTrie_;

	protected double [] xyCount;

	protected Instant[] timestamp;
	
	protected String adaptiveControl = "";
	protected double adaptiveControlParameter = 0;
	
	protected String experimentType = "";

	public wdAnDEParameters() throws FileNotFoundException, IOException {

		n = Globals.getNumAttributes();
		nc = Globals.getNumClasses();

		paramsPerAtt = Globals.getParamsPerAtt();
		isNumericTrue = Globals.getIsNumericTrue();
		
		adaptiveControl = Globals.getAdaptiveControl();
		adaptiveControlParameter = Globals.getAdaptiveControlParameter();
		
		experimentType = Globals.getExperimentType();

		level = Globals.getLevel();

		indexTrie_ = new indexTrie[n];				

		if (level == 0) {
			np = nc;
			for (int u1 = 0; u1 < n; u1++) {
				indexTrie_[u1] = new indexTrie();

				indexTrie_[u1].set(np);
				np += (paramsPerAtt[u1] * nc);
			}
		} else if (level == 1) {
			np = nc;
			for (int u1 = 0; u1 < n; u1++) {
				indexTrie_[u1] = new indexTrie();
				indexTrie_[u1].set(np);

				np += (paramsPerAtt[u1] * nc);

				indexTrie_[u1].children = new indexTrie[n];

				for (int u2 = 0; u2 < u1; u2++) {

					indexTrie_[u1].children[u2] = new indexTrie();
					indexTrie_[u1].children[u2].set(np);

					np += (paramsPerAtt[u1] * paramsPerAtt[u2] * nc);												
				}					
			}
		} else if (level == 2) {
			np = nc;
			for (int u1 = 0; u1 < n; u1++) {

				indexTrie_[u1] = new indexTrie();
				indexTrie_[u1].set(np);
				np += (paramsPerAtt[u1] * nc);

				indexTrie_[u1].children = new indexTrie[n];

				for (int u2 = 0; u2 < u1; u2++) {

					indexTrie_[u1].children[u2] = new indexTrie();
					indexTrie_[u1].children[u2].set(np);
					np += (paramsPerAtt[u1] * paramsPerAtt[u2] * nc);

					indexTrie_[u1].children[u2].children = new indexTrie[n];

					for (int u3 = 0; u3 < u2; u3++) {

						indexTrie_[u1].children[u2].children[u3] = new indexTrie();
						indexTrie_[u1].children[u2].children[u3].set(np);		

						np += (paramsPerAtt[u1] * paramsPerAtt[u2] * paramsPerAtt[u3] * nc);												
					}					
				}
			}
		} 
	}

	public abstract void updateFirstPass(Instance inst);

	public abstract void finishedFirstPass();

	public abstract boolean needSecondPass();

	public abstract void updateAfterFirstPass(Instance inst);

	public long getAttributeIndex(int att1, int att1val, int c) {
		long offset = indexTrie_[att1].offset;		
		return offset + c * (paramsPerAtt[att1]) + att1val;
	}

	public long getAttributeIndex(int att1, int att1val, int att2, int att2val, int c) {
		long offset = indexTrie_[att1].children[att2].offset;		
		return offset + c * (paramsPerAtt[att1] * paramsPerAtt[att2]) + 
				att2val * (paramsPerAtt[att1]) + 
				att1val;
	}

	public long getAttributeIndex(int att1, int att1val, int att2, int att2val, int att3, int att3val, int c) {
		long offset = indexTrie_[att1].children[att2].children[att3].offset;
		return offset + c * (paramsPerAtt[att1] * paramsPerAtt[att2] * paramsPerAtt[att3]) + 
				att3val * (paramsPerAtt[att1] * paramsPerAtt[att2]) + 
				att2val * (paramsPerAtt[att1]) + 
				att1val;
	}

	public long getAttributeIndex(int att1, int att1val, int att2, int att2val, int att3, int att3val, int att4, int att4val, int c) {
		long offset = indexTrie_[att1].children[att2].children[att3].children[att4].offset;
		return offset + c * (paramsPerAtt[att1] * paramsPerAtt[att2] * paramsPerAtt[att3] * paramsPerAtt[att4]) +
				att4val * (paramsPerAtt[att1] * paramsPerAtt[att2] * paramsPerAtt[att3]) +
				att3val * (paramsPerAtt[att1] * paramsPerAtt[att2]) + 
				att2val * (paramsPerAtt[att1]) + 
				att1val;
	}

	public long getAttributeIndex(int att1, int att1val, int att2, int att2val, int att3, int att3val, int att4, int att4val, int att5, int att5val, int c) {
		long offset = indexTrie_[att1].children[att2].children[att3].children[att4].children[att5].offset;		
		return offset + c * (paramsPerAtt[att1] * paramsPerAtt[att2] * paramsPerAtt[att3] * paramsPerAtt[att4] * paramsPerAtt[att5]) +
				att5val * (paramsPerAtt[att1] * paramsPerAtt[att2] * paramsPerAtt[att3] * paramsPerAtt[att4]) +
				att4val * (paramsPerAtt[att1] * paramsPerAtt[att2] * paramsPerAtt[att3]) +
				att3val * (paramsPerAtt[att1] * paramsPerAtt[att2]) + 
				att2val * (paramsPerAtt[att1]) + 
				att1val;
	}

	@Override
	public void initializeParametersWithVal(int val) {	
	}

	@Override
	public void convertToProbs() {
	}

	public abstract double getCountAtFullIndex(long index);

	public void allocateMemoryForCountsParametersProbabilitiesAndGradients(long size) {

		xyCount = new double[(int)size];

		setNp(size);

		if (experimentType.equalsIgnoreCase("prequential") || 
				experimentType.equalsIgnoreCase("flowMachines") || 
				experimentType.equalsIgnoreCase("drift")) {
			
			if (adaptiveControl.equalsIgnoreCase("Decay")) {
				
				timestamp = new Instant[(int) size];
				
				for (int i = 0; i < size; i++) {
					timestamp[i] = Instant.now();
				}
				
			}
		}

	}

} // ends class

