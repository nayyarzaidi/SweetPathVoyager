package utils;

import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;

import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomDataGenerator;
import org.apache.commons.math3.random.RandomGenerator;

import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;

import moa.core.FastVector;
import moa.core.InstanceExample;
import moa.core.ObjectRepository;
import moa.streams.InstanceStream;
import moa.tasks.TaskMonitor;

public class AbruptDriftGeneratorWithParameters extends DriftGenerator {

	private static final long serialVersionUID = 1291115908166720203L;

	protected InstancesHeader streamHeader;

	/**
	 * p(x) before drift
	 */
	double[][] pxbd;
	/**
	 * p(y|x) before drift
	 */
	double[][] pygxbd;

	/**
	 * p(x) after drift
	 */
	double[][] pxad;
	/**
	 * p(y|x) after drift
	 */
	double[][] pygxad;

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
		return true;
	}

	@Override
	public void restart() {
		nInstancesGeneratedSoFar = 0L;
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

		FastVector<Attribute> attributes = getHeaderAttributes(nAttributes
				.getValue(), nValuesPerAttribute.getValue());

		this.streamHeader = new InstancesHeader(new Instances(
				getCLICreationString(InstanceStream.class), attributes, 0));
		this.streamHeader.setClassIndex(this.streamHeader.numAttributes() - 1);
	}

	@Override
	public InstanceExample nextInstance() {
		double[][] px = (nInstancesGeneratedSoFar < burnInNInstances
				.getValue()) ? pxbd : pxad;
		double[][] pygx = (nInstancesGeneratedSoFar < burnInNInstances
				.getValue()) ? pygxbd : pygxad;

		Instance inst = new DenseInstance(streamHeader.numAttributes());
		inst.setDataset(streamHeader);

		int[] indexes = new int[nAttributes.getValue()];

		// setting values for x_1,...,x_n
		for (int a = 0; a < indexes.length; a++) {
			// choosing values of x_1,...,x_n
			double rand = r.nextUniform(0.0, 1.0, true);
			//double rand = srg.nextDouble();
			int chosenVal = 0;
			double sumProba = px[a][chosenVal];
			while (rand > sumProba) {
				chosenVal++;
				sumProba += px[a][chosenVal];
			}
			indexes[a] = chosenVal;
			inst.setValue(a, chosenVal);
		}

		int lineNoCPT = getIndex(indexes);

		double rand = r.nextUniform(0.0, 1.0,true);
		int chosenClassValue = 0;
		double sumProba = pygx[lineNoCPT][chosenClassValue];
		while (rand > sumProba) {
			chosenClassValue++;
			sumProba += pygx[lineNoCPT][chosenClassValue];
		}
		inst.setClassValue(chosenClassValue);

		nInstancesGeneratedSoFar++;
		// System.out.println("generated "+inst);
		return new InstanceExample(inst);
	}

	private SecureRandom srg = null;

	@Override
	protected void prepareForUseImpl(TaskMonitor monitor,
			ObjectRepository repository) {
		//        System.out.println("burnIn=" + burnInNInstances.getValue());
		generateHeader();

		RandomGenerator rg = new JDKRandomGenerator();
		try {
			srg = SecureRandom.getInstance("SHA1PRNG");
		} catch (NoSuchAlgorithmException e) {
			e.printStackTrace();
		}
		rg.setSeed(seed.getValue());

		r = new RandomDataGenerator(rg);

		// generating distribution before drift

		nInstancesGeneratedSoFar = 0L;

	}

	protected final int getIndex(int... indexes) {
		int index = indexes[0];
		for (int i = 1; i < indexes.length; i++) {
			index *= nValuesPerAttribute.getValue();
			index += indexes[i];
		}
		return index;

	}
	public void setPrioDistBeforeDrift(double[][]p){
		this.pxbd = p;
	}

	public void setPrioDistAfterDrift(double[][]p){
		this.pxad = p;
	}

	public void setCondDistAfterDrift(double[][]p){
		this.pygxad = p;
	}

	public void setCondDistBeforeDrift(double[][]p){
		this.pygxbd = p;
	}

}