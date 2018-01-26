package utils;

/*
 * Created by Lee on 6/09/2015.
 */

import moa.core.FastVector;
import moa.core.InstanceExample;
import moa.core.ObjectRepository;
import moa.options.AbstractOptionHandler;
import moa.streams.InstanceStream;
import moa.tasks.TaskMonitor;

import java.util.Arrays;

import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomDataGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.util.FastMath;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;

public class GradualDriftGeneratorLR extends DriftGenerator {
	private static final long serialVersionUID = 1291115908166720203L;
	public IntOption driftLength = new IntOption("driftLength", 'l', "The number of instances the concept drift will be applied acrros", 1000, 0,
					Integer.MAX_VALUE);
	protected final int nTrialsForGeneratingPYGX = 10000;
	protected InstancesHeader streamHeader;

	/**
	 * p(x) before drift
	 */
	double[][] pxbd;

	double betaInterceptBeforeDrift;
	double[][] betasFirstOrderBeforeDrift;
	double[][] betasSecondOrderBeforeDrift;

	/**
	 * p(x) after drift
	 */
	double[][] pxad;

	double betaInterceptAfterDrift;
	double[][] betasFirstOrderAfterDrift;
	double[][] betasSecondOrderAfterDrift;

	RandomDataGenerator r;

	long nInstancesGeneratedSoFar;

	@Override
	public long estimatedRemainingInstances() {
		return -1;
	}

	@Override
	public boolean hasMoreInstances() {
		return true;
	}

	@Override
	public boolean isRestartable() {
		return false;
	}

	@Override
	public void restart() {

	}

	@Override
	public void getDescription(StringBuilder sb, int indent) {

	}

	@Override
	public String getPurposeString() {
		return "Generates a stream with an abrupt drift of given magnitude.";
	}

	@Override
	public InstancesHeader getHeader() {
		return streamHeader;
	}

	protected void generateHeader() {

		FastVector<Attribute> attributes = getHeaderAttributes(nAttributes.getValue(), nValuesPerAttribute.getValue());

		this.streamHeader = new InstancesHeader(new Instances(getCLICreationString(InstanceStream.class), attributes, 0));
		this.streamHeader.setClassIndex(this.streamHeader.numAttributes() - 1);
	}

	@Override
	public InstanceExample nextInstance() {
		double[][] px;
		double betaIntercept;
		double[][] betasFirstOrder;
		double[][] betasSecondOrder;

		double probSecondDistrib;

		if (nInstancesGeneratedSoFar <= burnInNInstances.getValue()) {
			probSecondDistrib = 0.0;
		} else if (nInstancesGeneratedSoFar > burnInNInstances.getValue() + driftLength.getValue()) {
			probSecondDistrib = 1.0;
		} else {
			// prob proportional to where we're at in the drift
			int nInstancesInDrift = (int) (nInstancesGeneratedSoFar - burnInNInstances.getValue());
			probSecondDistrib = 1.0 * nInstancesInDrift / driftLength.getValue();
		}

		boolean isSecondDistrib = r.nextUniform(0.0, 1.0, true) <= probSecondDistrib;

		if (isSecondDistrib) {
			px = pxad;
			betaIntercept = betaInterceptAfterDrift;
			betasFirstOrder = betasFirstOrderAfterDrift;
			betasSecondOrder = betasSecondOrderAfterDrift;
		} else {
			px = pxbd;
			betaIntercept = betaInterceptBeforeDrift;
			betasFirstOrder = betasFirstOrderBeforeDrift;
			betasSecondOrder = betasSecondOrderBeforeDrift;
		}

		Instance inst = new DenseInstance(streamHeader.numAttributes());
		inst.setDataset(streamHeader);

		int[] indexes = new int[nAttributes.getValue()];

		// System.out.println("Setting Values for x_n");
		// setting values for x_1,...,x_n
		for (int a = 0; a < indexes.length; a++) {
			// System.out.println("a: " + a);
			// choosing values of x_1,...,x_n
			double rand = r.nextUniform(0.0, 1.0, true);
			int chosenVal = 0;
			double sumProba = px[a][chosenVal];
			while (rand > sumProba) {
				// System.out.println("class val: " +
				// chosenVal);
				chosenVal++;
				sumProba += px[a][chosenVal];
			}
			indexes[a] = chosenVal;
			inst.setValue(a, chosenVal);
		}

		// now setting y as LR with beta^T.x
		double py = calculatePy(betaIntercept, betasFirstOrder, betasSecondOrder, indexes);
//		System.out.println(py);
		// sampling y
		int y = (r.nextUniform(0.0, 1.0) < py) ? 1 : 0;
		inst.setClassValue(y);

		nInstancesGeneratedSoFar++;
		// System.out.println("generated "+inst);
		return new InstanceExample(inst);
	}

	public static void generateRandomPyGivenX(double[][] betasFirstOrder, double[][] betasSecondOrder, RandomDataGenerator r) {

		for (int a = 0; a < betasFirstOrder.length; a++) {
			for (int v = 0; v < betasFirstOrder[a].length; v++) {
				do{
					betasFirstOrder[a][v] = r.nextGaussian(0.0, 1.0);
				}while(Math.abs(betasFirstOrder[a][v])>3.0);
				
			}
		}

		for (int a = 0; a < betasSecondOrder.length; a++) {
			for (int v = 0; v < betasSecondOrder[a].length; v++) {
				do{
					betasSecondOrder[a][v] = r.nextGaussian(0.0, 1.0);
				}while(Math.abs(betasSecondOrder[a][v])>3.0);
			}
		}
		
		

	}
	
	public static double generateRandomPyGivenXFlatPrior(int nCombinationsValuesForPX,double[][]px,double[][] betasFirstOrder, double[][] betasSecondOrder, RandomDataGenerator r) {
		
		
		for (int a = 0; a < betasFirstOrder.length; a++) {
			for (int v = 0; v < betasFirstOrder[a].length; v++) {
				do{
					betasFirstOrder[a][v] = r.nextGaussian(0.0, 1.0);
				}while(Math.abs(betasFirstOrder[a][v])>3.0);
				
			}
		}

		for (int a = 0; a < betasSecondOrder.length; a++) {
			for (int v = 0; v < betasSecondOrder[a].length; v++) {
				do{
					betasSecondOrder[a][v] = r.nextGaussian(0.0, 1.0);
				}while(Math.abs(betasSecondOrder[a][v])>3.0);
			}
		}
		
		double intercept = r.nextGaussian(0.0, 1.0);
		//adjust intercept such that priors for class are similar
		double precision = 0.001;
		double targetPrior = 0.5;
		double prior = calculateClassPrior(nCombinationsValuesForPX, px, intercept, betasFirstOrder, betasSecondOrder);
		while(Math.abs(prior-targetPrior)>precision){
			if(prior>targetPrior){
				intercept-=0.001;
			}else{
				intercept+=0.001;
			}
			prior = calculateClassPrior(nCombinationsValuesForPX, px, intercept, betasFirstOrder, betasSecondOrder);
//			System.out.println("prior before = "+priorBefore+"\tprior after="+priorAfter);
		}
//		System.out.println("managed to initialise the distribution to get prior at "+prior);
		return intercept;
		

	}
	
	
	
	

	@Override
	protected void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
		System.out.println("burnIn=" + burnInNInstances.getValue());
		generateHeader();

		int nCombinationsValuesForPX = 1;
		for (int a = 0; a < nAttributes.getValue(); a++) {
			nCombinationsValuesForPX *= nValuesPerAttribute.getValue();
		}

		pxbd = new double[nAttributes.getValue()][nValuesPerAttribute.getValue()];
		betasFirstOrderBeforeDrift = new double[nAttributes.getValue()][nValuesPerAttribute.getValue()];
		betasSecondOrderBeforeDrift = new double[nAttributes.getValue() - 1][];
		for (int a = 0; a < betasSecondOrderBeforeDrift.length; a++) {
			betasSecondOrderBeforeDrift[a] = new double[pxbd[a].length * pxbd[a + 1].length];
		}

		RandomGenerator rg = new JDKRandomGenerator();
		rg.setSeed(seed.getValue());
		r = new RandomDataGenerator(rg);

		// generating distribution before drift

		// p(x)
		generateRandomPx(pxbd, r);

		// generating distribution after drift

		if (driftPriors.isSet()) {
			pxad = new double[nAttributes.getValue()][nValuesPerAttribute.getValue()];

			System.out.println("Sampling p(x) for required magnitude...");
			double sigma = 0.001;
			generateRandomPxAfterCloseToBefore(sigma, pxbd, pxad, r);
			double obtainedMagnitude = computeMagnitudePX(nCombinationsValuesForPX, pxbd, pxad);
			boolean haveReducedSigma = true;
			double coef = 2.0;

			while (Math.abs(obtainedMagnitude - driftMagnitudePrior.getValue()) > precisionDriftMagnitude.getValue()) {

				if (obtainedMagnitude > driftMagnitudePrior.getValue()) {
					if (!haveReducedSigma) {
						// had just increased it, now
						// decreasing, so making the
						// coefficient smaller
						coef = 1.0 + coef / 2.0;
					}
					sigma /= coef;
					haveReducedSigma = true;
				} else {
					if (haveReducedSigma) {
						// had just decreased it, now
						// increasing, so making the
						// coefficient smaller
						coef = 1.0 + coef / 2.0;
					}
					sigma *= coef;
					haveReducedSigma = false;
				}

				generateRandomPxAfterCloseToBefore(sigma, pxbd, pxad, r);
				obtainedMagnitude = computeMagnitudePX(nCombinationsValuesForPX, pxbd, pxad);
				
			}
			
			System.out.println(sigma);
			System.out.println("exact magnitude for p(x)=" + computeMagnitudePX(nCombinationsValuesForPX, pxbd, pxad) + "\tasked="
							+ driftMagnitudePrior.getValue());
		} else {
			pxad = pxbd;
		}

		// p(y|x)
//		double intercept = r.nextGaussian(0.0, 1.0);
//		betaInterceptBeforeDrift = intercept;
//		generateRandomPyGivenX(betasFirstOrderBeforeDrift, betasSecondOrderBeforeDrift, r);
		betaInterceptBeforeDrift = generateRandomPyGivenXFlatPrior(nCombinationsValuesForPX,pxbd,betasFirstOrderBeforeDrift, betasSecondOrderBeforeDrift, r);
		betaInterceptAfterDrift = betaInterceptBeforeDrift;

		// conditional
		if (driftConditional.isSet()) {

			betasFirstOrderAfterDrift = new double[nAttributes.getValue()][nValuesPerAttribute.getValue()];
			betasSecondOrderAfterDrift = new double[nAttributes.getValue() - 1][];
			for (int a = 0; a < betasSecondOrderAfterDrift.length; a++) {
				betasSecondOrderAfterDrift[a] = new double[pxbd[a].length * pxbd[a + 1].length];
			}
			
//			double[][]tmp1= new double[nAttributes.getValue()][nValuesPerAttribute.getValue()];
//			double[][]tmp2= new double[nAttributes.getValue() - 1][];
//			for (int a = 0; a < tmp2.length; a++) {
//				tmp2[a] = new double[pxbd[a].length * pxbd[a + 1].length];
//			}
//			for (int a = 0; a < betasFirstOrderBeforeDrift.length; a++) {
//				for (int v = 0; v < betasFirstOrderBeforeDrift[a].length; v++) {
//					tmp1[a][v] = betasFirstOrderBeforeDrift[a][v];
//					betasFirstOrderAfterDrift[a][v] = betasFirstOrderBeforeDrift[a][v];
//				}
//			}
//
//			for (int a = 0; a < betasSecondOrderBeforeDrift.length; a++) {
//				for (int i = 0; i < betasSecondOrderBeforeDrift[a].length; i++) {
//					tmp2[a][i] = betasSecondOrderBeforeDrift[a][i];
//					betasSecondOrderAfterDrift[a][i] = betasSecondOrderBeforeDrift[a][i];
//				}
//			}
//			double[][]bestBetasFirstAfter= new double[nAttributes.getValue()][nValuesPerAttribute.getValue()];
//			double[][]bestBetasSecondAfter= new double[nAttributes.getValue() - 1][];
//			for (int a = 0; a < bestBetasSecondAfter.length; a++) {
//				bestBetasSecondAfter[a] = new double[pxbd[a].length * pxbd[a + 1].length];
//			}
//			
//			double bestMagDiff = 2.0; 
			
//			
//			
//			for(int nTrials=0;nTrials<nTrialsForGeneratingPYGX;nTrials++){
//				
//				swapCoefficients(tmp1,tmp2,betasFirstOrderAfterDrift,betasSecondOrderAfterDrift,r);
//				
//				double obtainedMagnitude = computeMagnitudePYGXLR(nCombinationsValuesForPX);
//				System.out.println(obtainedMagnitude);
//				double magDiff = Math.abs(obtainedMagnitude-driftMagnitudeConditional.getValue());
//				
//				if(magDiff<bestMagDiff){
//					for (int a = 0; a < betasFirstOrderBeforeDrift.length; a++) {
//						for (int v = 0; v < betasFirstOrderBeforeDrift[a].length; v++) {
//							bestBetasFirstAfter[a][v] = betasFirstOrderAfterDrift[a][v];
//						}
//					}
//
//					for (int a = 0; a < betasSecondOrderBeforeDrift.length; a++) {
//						for (int i = 0; i < betasSecondOrderBeforeDrift[a].length; i++) {
//							bestBetasSecondAfter[a][i] = betasSecondOrderAfterDrift[a][i];
//						}
//					}
//					bestMagDiff = magDiff;
//					if(magDiff<= precisionDriftMagnitude.getValue()){
//						break;
//					}
//				}
//				
//				if(nTrials%100==0){
//					for (int a = 0; a < betasFirstOrderBeforeDrift.length; a++) {
//						for (int v = 0; v < betasFirstOrderBeforeDrift[a].length; v++) {
//							tmp1[a][v] = bestBetasFirstAfter[a][v];
//						}
//					}
//
//					for (int a = 0; a < betasSecondOrderBeforeDrift.length; a++) {
//						for (int i = 0; i < betasSecondOrderBeforeDrift[a].length; i++) {
//							tmp2[a][i] = bestBetasSecondAfter[a][i];
//						}
//					}
//				}
//				
//			}
//			

			double sigma = 0.1;
//			generateRandomPyGxAfterCloseToBeforeLR(sigma, r);
//			if(driftMagnitudeConditional.getValue()>0.1){
				generateRandomPyGxAfterSameClassPriorLR(r,nCombinationsValuesForPX);
//			}else{
//				generateRandomPyGxAfterCloseToBeforeSameClassPriorLR(sigma, r,nCombinationsValuesForPX);
//			}
//			generateRandomPyGivenX(betasFirstOrderAfterDrift, betasSecondOrderAfterDrift, r);
			double obtainedMagnitude = computeMagnitudePYGXLRWeighted(nCombinationsValuesForPX);

			boolean haveReducedSigma = true;
			double coef = 2.0;
			int nTrials = 0;
			double bestMagDiff = 2.0;
			
			double bestBetaIntercept=betaInterceptAfterDrift;
			double[][]bestBetasFirstAfter= new double[nAttributes.getValue()][nValuesPerAttribute.getValue()];
			double[][]bestBetasSecondAfter= new double[nAttributes.getValue() - 1][];
			for (int a = 0; a < bestBetasSecondAfter.length; a++) {
				bestBetasSecondAfter[a] = new double[pxbd[a].length * pxbd[a + 1].length];
			}
			
			while (nTrials<nTrialsForGeneratingPYGX  && Math.abs(obtainedMagnitude - driftMagnitudeConditional.getValue()) > precisionDriftMagnitude.getValue()) {
				
				if (obtainedMagnitude > driftMagnitudeConditional.getValue()) {
					if (!haveReducedSigma) {
						// had just increased it, now
						// decreasing, so making the
						// coefficient smaller
						coef = 1.0 + coef / 2.0;
					}
					sigma /= coef;
					sigma = Math.max(sigma, 10e-5);
					haveReducedSigma = true;
				} else {
					if (haveReducedSigma) {
						// had just decreased it, now
						// increasing, so making the
						// coefficient smaller
						coef = 1.0 + coef / 2.0;
					}
					sigma *= coef;
					sigma = Math.min(sigma, 2);
					haveReducedSigma = false;
				}

//				generateRandomPyGxAfterCloseToBeforeLR(sigma, r);
//				generateRandomPyGxAfterCloseToBeforeSameClassPriorLR(sigma, r,nCombinationsValuesForPX);
//				if(driftMagnitudeConditional.getValue()>0.1){
					generateRandomPyGxAfterSameClassPriorLR(r,nCombinationsValuesForPX);
//				}else{
//					generateRandomPyGxAfterCloseToBeforeSameClassPriorLR(sigma, r,nCombinationsValuesForPX);
//				}
//				generateRandomPyGivenX(betasFirstOrderAfterDrift, betasSecondOrderAfterDrift, r);
				obtainedMagnitude = computeMagnitudePYGXLRWeighted(nCombinationsValuesForPX);
				
				double magDiff = Math.abs(obtainedMagnitude-driftMagnitudeConditional.getValue());
				if(magDiff<bestMagDiff){
					bestBetaIntercept = betaInterceptAfterDrift;
					for (int a = 0; a < betasFirstOrderBeforeDrift.length; a++) {
						for (int v = 0; v < betasFirstOrderBeforeDrift[a].length; v++) {
							bestBetasFirstAfter[a][v] = betasFirstOrderAfterDrift[a][v];
						}
					}

					for (int a = 0; a < betasSecondOrderBeforeDrift.length; a++) {
						for (int i = 0; i < betasSecondOrderBeforeDrift[a].length; i++) {
							bestBetasSecondAfter[a][i] = betasSecondOrderAfterDrift[a][i];
						}
					}
					bestMagDiff = magDiff;
				}
//				System.out.println("asked m="+driftMagnitudeConditional.getValue()+"\tobtained="+obtainedMagnitude);
				
				nTrials++;
			}
			if(nTrials==nTrialsForGeneratingPYGX){
				System.out.println("Warning, didn't manage to generate the requested magnitude");
				betaInterceptAfterDrift = bestBetaIntercept; 
				betasFirstOrderAfterDrift=bestBetasFirstAfter ;
				 betasSecondOrderAfterDrift=bestBetasSecondAfter;
			}
			System.out.println("exact magnitude for p(y|x)=" + computeMagnitudePYGXLRWeighted(nCombinationsValuesForPX) + "\tasked="
							+ driftMagnitudeConditional.getValue());
		} else {
			betasFirstOrderAfterDrift = betasFirstOrderBeforeDrift;
			betasSecondOrderAfterDrift = betasSecondOrderBeforeDrift;
		}
		
		System.out.println("prior class before: "+calculateClassPrior(nCombinationsValuesForPX, pxbd, betaInterceptBeforeDrift, betasFirstOrderBeforeDrift, betasSecondOrderBeforeDrift));
		System.out.println("prior class after: "+calculateClassPrior(nCombinationsValuesForPX, pxad, betaInterceptAfterDrift, betasFirstOrderAfterDrift, betasSecondOrderAfterDrift));
		System.out.println("intercept before: "+betaInterceptBeforeDrift);
		System.out.println("intercept after: "+betaInterceptAfterDrift);
		for (int a = 0; a < betasFirstOrderBeforeDrift.length; a++) {
			System.out.println("first order before: "+Arrays.toString(betasFirstOrderBeforeDrift[a]));
			System.out.println("first order after: "+ Arrays.toString(betasFirstOrderAfterDrift[a]));
		}
		for (int a = 0; a < betasFirstOrderBeforeDrift.length-1; a++) {
			System.out.println("second order before: "+Arrays.toString(betasSecondOrderBeforeDrift[a]));
			System.out.println("second order after: "+Arrays.toString(betasSecondOrderAfterDrift[a]));
		}
		System.out.println();

		// System.out.println(Arrays.toString(pxbd));
		// System.out.println(Arrays.toString(pxad));

		nInstancesGeneratedSoFar = 0L;

	}

	private void swapCoefficients(double[][] tmp1, double[][] tmp2,
			double[][] betasFirstOrderAfterDrift2, double[][] betasSecondOrderAfterDrift2,RandomDataGenerator r) {
		int nAttributes = betasFirstOrderAfterDrift2.length;
		int a1 = r.nextInt(0, nAttributes-1);
		int a2;
		do{
			a2 = r.nextInt(0, nAttributes-1);
		}while(a1==a2);
		
		for (int a = 0; a < betasFirstOrderBeforeDrift.length; a++) {
			int tmpA;
			if(a==a1){
				tmpA=a2;
			}else if (a==a2){
				tmpA=a1;
			}else{
				tmpA=a;
			}
			for (int v = 0; v < betasFirstOrderBeforeDrift[a].length; v++) {
				betasFirstOrderAfterDrift2[a][v] = tmp1[tmpA][v];
			}
		}
		
	}

	protected final int getIndex(int... indexes) {
		int index = indexes[0];
		for (int i = 1; i < indexes.length; i++) {
			index *= nValuesPerAttribute.getValue();
			index += indexes[i];
		}
		return index;

	}

	private void generateRandomPyGxAfterCloseToBeforeLR(double sigma, RandomDataGenerator r2) {
		for (int a = 0; a < betasFirstOrderBeforeDrift.length; a++) {
			for (int v = 0; v < betasFirstOrderBeforeDrift[a].length; v++) {
				do{
					betasFirstOrderAfterDrift[a][v] = r2.nextGaussian(betasFirstOrderBeforeDrift[a][v], sigma);
				}while(Math.abs(betasFirstOrderAfterDrift[a][v])>2.0);
				
			}
		}

		for (int a = 0; a < betasSecondOrderBeforeDrift.length; a++) {
			for (int v = 0; v < betasSecondOrderBeforeDrift[a].length; v++) {
				do{
					betasSecondOrderAfterDrift[a][v] = r2.nextGaussian(betasSecondOrderBeforeDrift[a][v], sigma);
				}while(Math.abs(betasSecondOrderAfterDrift[a][v])>2.0);
//				if(Double.isInfinite(betasSecondOrderAfterDrift[a][i])){
//					System.out.println(betasSecondOrderBeforeDrift[a][i]+"\t"+sigma);
//					System.exit(0);
//				}
			}
		}

	}
	
	private void generateRandomPyGxAfterCloseToBeforeSameClassPriorLR(double sigma, RandomDataGenerator r2,int nCombinationsValuesForPX) {
		
		//calculate prior before
		if(driftPriors.isSet())throw new RuntimeException("Shouldn't use generateRandomPyGxAfterCloseToBeforeSameClassPriorLR if prior is being drifted");
		
		double m = 0.0;
		double priorBefore = calculateClassPrior(nCombinationsValuesForPX, pxbd, betaInterceptBeforeDrift, betasFirstOrderBeforeDrift, betasSecondOrderBeforeDrift);
		
		//change distribution for first and second order terms		
		for (int a = 0; a < betasFirstOrderBeforeDrift.length; a++) {
			for (int v = 0; v < betasFirstOrderBeforeDrift[a].length; v++) {
				do{
					betasFirstOrderAfterDrift[a][v] = r2.nextGaussian(betasFirstOrderBeforeDrift[a][v], sigma);
				}while(Math.abs(betasFirstOrderAfterDrift[a][v])>2.0);
				
			}
		}

		for (int a = 0; a < betasSecondOrderBeforeDrift.length; a++) {
			for (int v = 0; v < betasSecondOrderBeforeDrift[a].length; v++) {
				do{
					betasSecondOrderAfterDrift[a][v] = r2.nextGaussian(betasSecondOrderBeforeDrift[a][v], sigma);
				}while(Math.abs(betasSecondOrderAfterDrift[a][v])>2.0);
//				if(Double.isInfinite(betasSecondOrderAfterDrift[a][i])){
//					System.out.println(betasSecondOrderBeforeDrift[a][i]+"\t"+sigma);
//					System.exit(0);
//				}
			}
		}
		
		
		//adjust intercept such that priors for class are similar
		double precision = 0.01;
		double priorAfter = calculateClassPrior(nCombinationsValuesForPX, pxad, betaInterceptAfterDrift, betasFirstOrderAfterDrift, betasSecondOrderAfterDrift);
		while(Math.abs(priorAfter-priorBefore)>precision){
			if(priorAfter>priorBefore){
				betaInterceptAfterDrift-=0.01;
			}else{
				betaInterceptAfterDrift+=0.01;
			}
			priorAfter = calculateClassPrior(nCombinationsValuesForPX, pxad, betaInterceptAfterDrift, betasFirstOrderAfterDrift, betasSecondOrderAfterDrift);
//			System.out.println("prior before = "+priorBefore+"\tprior after="+priorAfter);
		}
//		System.out.println("managed to reestablish prior\tprior before = "+priorBefore+"\tprior after="+priorAfter);
	}
	
	private void generateRandomPyGxAfterSameClassPriorLR( RandomDataGenerator r2,int nCombinationsValuesForPX) {
		
		//calculate prior before
		double priorBefore = calculateClassPrior(nCombinationsValuesForPX, pxbd, betaInterceptBeforeDrift, betasFirstOrderBeforeDrift, betasSecondOrderBeforeDrift);
		
		generateRandomPyGivenX(betasFirstOrderAfterDrift, betasSecondOrderAfterDrift, r2);
		
		//adjust intercept such that priors for class are similar
		double precision = 0.001;
		double priorAfter = calculateClassPrior(nCombinationsValuesForPX, pxad, betaInterceptAfterDrift, betasFirstOrderAfterDrift, betasSecondOrderAfterDrift);
		while(Math.abs(priorAfter-priorBefore)>precision){
			if(priorAfter>priorBefore){
				betaInterceptAfterDrift-=0.001;
			}else{
				betaInterceptAfterDrift+=0.001;
			}
			priorAfter = calculateClassPrior(nCombinationsValuesForPX, pxad, betaInterceptAfterDrift, betasFirstOrderAfterDrift, betasSecondOrderAfterDrift);
//			System.out.println("prior before = "+priorBefore+"\tprior after="+priorAfter);
		}
//		System.out.println("managed to reestablish prior\tprior before = "+priorBefore+"\tprior after="+priorAfter);
	}
	
	private static double calculateClassPrior(int nCombinationsValuesForPX,double[][]px,double betaIntercept,double[][]betasFirstOrder,double[][]betasSecondOrder){
		double prior = 0.0;
		int[] indexes = new int[px.length];
		for (int i = 0; i < nCombinationsValuesForPX; i++) {
			getIndexes(i, indexes, px[0].length);
			
			double tmpPrior = 1.0;
			for(int a=0;a<indexes.length;a++){
				tmpPrior *= px[a][indexes[a]];
			}
			
			double py = calculatePy(betaIntercept, betasFirstOrder, betasSecondOrder, indexes);
			prior += tmpPrior*py;
		}
		return prior;
	}

	public double computeMagnitudePYGXLR(int nCombinationsValuesForPX) {

		int[] indexes = new int[pxbd.length];
		double m = 0.0;
		for (int i = 0; i < nCombinationsValuesForPX; i++) {
			getIndexes(i, indexes, nValuesPerAttribute.getValue());

			double pyBefore = calculatePy(betaInterceptBeforeDrift, betasFirstOrderBeforeDrift, betasSecondOrderBeforeDrift, indexes);
			double pyAfter = calculatePy(betaInterceptAfterDrift, betasFirstOrderAfterDrift, betasSecondOrderAfterDrift, indexes);
			double partialM = 0.0;
			// class = 1
			double diff = Math.sqrt(pyBefore) - Math.sqrt(pyAfter);
			partialM += diff * diff;

			// class = 0
			diff = Math.sqrt(1.0 - pyBefore) - Math.sqrt(1.0 - pyAfter);
			partialM += diff * diff;

			partialM = Math.sqrt(partialM) / Math.sqrt(2);
			m += partialM;

		}
		m /= nCombinationsValuesForPX;
		return m;
	}
	
	/**
	 * Computes the magnitude of the drift for the posterior, weighted by the probability of each combination of values in px. 
	 * This assumes that px is not drifted
	 * @param nCombinationsValuesForPX
	 * @return
	 */
	public double computeMagnitudePYGXLRWeighted(int nCombinationsValuesForPX) {
		if(driftPriors.isSet())throw new RuntimeException("Shouldn't use computeMagnitudePYGXLRWeighted if prior is being drifted");
		int[] indexes = new int[pxbd.length];
		double m = 0.0;
		for (int i = 0; i < nCombinationsValuesForPX; i++) {
			getIndexes(i, indexes, nValuesPerAttribute.getValue());
			
			double px = 1.0;
			for(int a=0;a<indexes.length;a++){
				px *= pxbd[a][indexes[a]];
			}
			
			double pyBefore = calculatePy(betaInterceptBeforeDrift, betasFirstOrderBeforeDrift, betasSecondOrderBeforeDrift, indexes);
			double pyAfter = calculatePy(betaInterceptAfterDrift, betasFirstOrderAfterDrift, betasSecondOrderAfterDrift, indexes);
			double partialM = 0.0;
			// class = 1
			double diff = Math.sqrt(pyBefore) - Math.sqrt(pyAfter);
			partialM += diff * diff;

			// class = 0
			diff = Math.sqrt(1.0 - pyBefore) - Math.sqrt(1.0 - pyAfter);
			partialM += diff * diff;

			partialM = Math.sqrt(partialM) / Math.sqrt(2);
			m += px*partialM;

		}
		
		return m;
	}

	public static double calculatePy(double intercept, double[][] betasFirstOrder, double[][] betasSecondOrder, int[] instanceValuesIndexes) {
		double py = intercept;

		// first-order terms
		for (int a = 0; a < instanceValuesIndexes.length; a++) {
			py += betasFirstOrder[a][instanceValuesIndexes[a]];
		}

		// second-order terms (assumes that the model includes x1-x2,
		// x2-x3, x3-x4 ...
		for (int a = 0; a < instanceValuesIndexes.length - 1; a++) {
			int index = instanceValuesIndexes[a] * betasFirstOrder[a + 1].length + instanceValuesIndexes[a + 1];
			double beta = betasSecondOrder[a][index];
			py += beta;
		}
		// apply logistic function
		py = 1.0 / (1.0 + FastMath.exp(-py));
		
		if(Double.isNaN(py)){
			for (int a = 0; a < instanceValuesIndexes.length; a++) {
				System.out.println(Arrays.toString(betasFirstOrder[a]));
			}
			for (int a = 0; a < instanceValuesIndexes.length - 1; a++) {
				System.out.println(Arrays.toString(betasSecondOrder[a]));
			}
		}
		return py;
		
		
	}
	
	public double[][] getPxbd() {
		return pxbd;
	}

	public void setPxbd(double[][] pxbd) {
		this.pxbd = pxbd;
	}

	public double getBetaInterceptBeforeDrift() {
		return betaInterceptBeforeDrift;
	}

	public void setBetaInterceptBeforeDrift(double betaInterceptBeforeDrift) {
		this.betaInterceptBeforeDrift = betaInterceptBeforeDrift;
	}

	public double[][] getBetasFirstOrderBeforeDrift() {
		return betasFirstOrderBeforeDrift;
	}

	public void setBetasFirstOrderBeforeDrift(double[][] betasFirstOrderBeforeDrift) {
		this.betasFirstOrderBeforeDrift = betasFirstOrderBeforeDrift;
	}

	public double[][] getBetasSecondOrderBeforeDrift() {
		return betasSecondOrderBeforeDrift;
	}

	public void setBetasSecondOrderBeforeDrift(double[][] betasSecondOrderBeforeDrift) {
		this.betasSecondOrderBeforeDrift = betasSecondOrderBeforeDrift;
	}

	public double[][] getPxad() {
		return pxad;
	}

	public void setPxad(double[][] pxad) {
		this.pxad = pxad;
	}

	public double getBetaInterceptAfterDrift() {
		return betaInterceptAfterDrift;
	}

	public void setBetaInterceptAfterDrift(double betaInterceptAfterDrift) {
		this.betaInterceptAfterDrift = betaInterceptAfterDrift;
	}

	public double[][] getBetasFirstOrderAfterDrift() {
		return betasFirstOrderAfterDrift;
	}

	public void setBetasFirstOrderAfterDrift(double[][] betasFirstOrderAfterDrift) {
		this.betasFirstOrderAfterDrift = betasFirstOrderAfterDrift;
	}

	public double[][] getBetasSecondOrderAfterDrift() {
		return betasSecondOrderAfterDrift;
	}

	public void setBetasSecondOrderAfterDrift(double[][] betasSecondOrderAfterDrift) {
		this.betasSecondOrderAfterDrift = betasSecondOrderAfterDrift;
	}

	public RandomDataGenerator getR() {
		return r;
	}

	public void setR(RandomDataGenerator r) {
		this.r = r;
	}
	
	
}
