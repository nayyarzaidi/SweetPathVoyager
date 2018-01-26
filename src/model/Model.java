package model;

import java.io.File;
import java.io.IOException;

import weka.core.Instance;
import weka.core.Instances;

public abstract class Model {
	
	public abstract void buildClassifier() throws Exception;
	
	public abstract double evaluateFunction(File sourceFile) throws IOException;
	
	public abstract double[] predict(Instance instance);
	public abstract double[] distributionForInstance(Instance instance);
	
	public abstract void computeGrad(Instance inst, double[] probs, int x_C);
	public abstract void computeGradAndUpdateParameters(Instance instance, double[] probs, int x_C);

	public abstract double[] evaluateFunction(Instances cvInstances);

	public abstract void update(Instance row);
	
}
