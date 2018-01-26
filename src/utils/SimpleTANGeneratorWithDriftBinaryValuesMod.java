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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomDataGenerator;
import org.apache.commons.math3.random.RandomGenerator;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;

public class SimpleTANGeneratorWithDriftBinaryValuesMod extends DriftGenerator {

	private static final long serialVersionUID = 1291115908166720203L;

	public IntOption driftLength = new IntOption("driftLength", 'l', "The number of instances the concept drift will be applied acrros", 0, 0, 10000000);
	public IntOption frequency = new IntOption("frequency", 'F', "The number of instances after which to change an entry in pygxbd", 0, 0, 100000);

	protected InstancesHeader streamHeader;

	double[] p_y;
	
	double[][][] p_yx;
	double[][][][] p_yxx;

	int d_yx[][];  // direction for yx - initialise randomly to 0 or 1
	int d_yxx[][][];  // direction for yxx - initialise randomly to 0 or 1

	int[] parents;

	int[] randAttributes;

	RandomDataGenerator r;

	long nInstancesGeneratedSoFar;

	int m_Q;

	private double delta = Globals.getDriftDelta();

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

		//FastVector<Attribute> attributes = getHeaderAttributes(nAttributes.getValue(), nValuesPerAttribute.getValue());

		int n = nAttributes.getValue() ;
		int nvals = 2;
		m_Q =  (int) (Globals.getDriftMagnitude3() * (n-1));

		System.out.println("Out of " + n + " attributes, " + (m_Q + 1) + " of them will be order 1 and " + (n - 1 - m_Q)  + " will be order 2");

		FastVector<Attribute> attributes = new FastVector<>();
		List<String> attributeValues = new ArrayList<String>();
		for (int v = 0; v < nvals; v++) {
			attributeValues.add("v" + (v + 1));
		}
		for (int i = 0; i < n; i++) {
			attributes.addElement(new Attribute("x" + (i + 1), attributeValues));
		}
		List<String> classValues = new ArrayList<String>();

		for (int v = 0; v < Globals.getNumClasses(); v++) {
			classValues.add("class" + (v + 1));
		}
		attributes.addElement(new Attribute("class", classValues));

		this.streamHeader = new InstancesHeader(new Instances(getCLICreationString(InstanceStream.class), attributes, 0));
		this.streamHeader.setClassIndex(this.streamHeader.numAttributes() - 1);
	}

	@Override
	public InstanceExample nextInstance() {

		int n = streamHeader.numAttributes() - 1;

		int numDriftAttributes = (int)Globals.getDriftMagnitude();

		if (numDriftAttributes > n) {
			System.out.println("Number of attributes with drift can't be greater than actual number of attributes");
			System.exit(-1);
		}

		if (Globals.getDriftMagnitude2() != 0) {
			if (nInstancesGeneratedSoFar % Globals.getDriftMagnitude2() == 0) {

				//intializeCPTBinary();

				/* Start: Introduce some Posterior-drift before generating the data */
				randAttributes = new int[numDriftAttributes]; 
				for (int i = 0; i < randAttributes.length; i++) {
					randAttributes[i] = -1;
				}

				int size = 0;
				while (size < numDriftAttributes) {
					int p = r.nextInt(1, n-1);

					if (!SUtils.inArray(p, randAttributes)) {
						randAttributes[size] = p;
						size++;
					}
				}

				//System.out.println("Rand Attributes to be drifted are " + Arrays.toString(randAttributes));
			}
		}
	
		for (int i = 0; i < numDriftAttributes; i++) {
			int p = randAttributes[i];
		

			if (p == 0) {
				/* Introduce drift in p_yx */
				//intoducePosteriorDrift(p_yx, delta);

			} else {
				if (p < m_Q ) {
					intoducePosteriorDrift(p_yx[p], d_yx[p], delta);	
				} else {
					/* Introduce drift in p_yxx */
					intoducePosteriorDrift(p_yxx[p], d_yxx[p], delta);
				}
			}
		}

		/* End: Introduce some Posterior-drift before generating the data */

		Instance inst = new DenseInstance(streamHeader.numAttributes());
		inst.setDataset(streamHeader);

		int nc = Globals.getNumClasses();

		int y = r.nextInt(0, nc - 1);
		inst.setClassValue(y);

		int[] x = new int[n];
		Arrays.fill(x, -1);

		x[0] = SUtils.sampleFromNonUniformDistribution(p_yx[0][y], r);

		inst.setValue(0, x[0]);

		for (int i = 1; i < n;  i++) {
			int p = parents[i];
			int xp = x[p];

			if (i < m_Q ) {
				x[i] = SUtils.sampleFromNonUniformDistribution(p_yx[i][y], r);	
			} else {
				x[i] = SUtils.sampleFromNonUniformDistribution(p_yxx[i][y][xp], r);
			}

			inst.setValue(i, x[i]);
		}

		nInstancesGeneratedSoFar++;

		return new InstanceExample(inst);
	}

	private void intoducePosteriorDrift(double p_yx[][], int d_yx[], double delta) {

		int y;  // the class

		for (y = 0; y <= 1; y++) {
			if (p_yx[y][0] < delta)  {
				d_yx[y] = 1;
				p_yx[y][0] += delta;
				p_yx[y][1] = 1.0 - p_yx[y][0];
			} else if (p_yx[y][0] > 1.0 - delta)  {
				d_yx[y] = 0;
				p_yx[y][0] -= delta;
				p_yx[y][1] = 1.0 - p_yx[y][0];
			} else if (d_yx[y] == 1)  {
				p_yx[y][0] += delta;
				p_yx[y][1] = 1.0 - p_yx[y][0];
			} else {
				p_yx[y][0] -= delta;
				p_yx[y][1] = 1.0 - p_yx[y][0];
			}
		}

	}

	private void intoducePosteriorDrift(double p_yxx[][][], int d_yxx[][], double delta) {

		int p;  // the parent
		int y;  // the class

		for (p = 0; p <= 1; p++) {
			for (y = 0; y <= 1; y++) {
				if (p_yxx[y][p][0] < delta)  {
					d_yxx[y][p] = 1;
					p_yxx[y][p][0] += delta;
					p_yxx[y][p][1] = 1.0 - p_yxx[y][p][0];
				} else if (p_yxx[y][p][0] > 1.0 - delta)  {
					d_yxx[y][p] = 0;
					p_yxx[y][p][0] -= delta;
					p_yxx[y][p][1] = 1.0 - p_yxx[y][p][0];
				} else if (d_yxx[y][p] == 1)  {
					p_yxx[y][p][0] += delta;
					p_yxx[y][p][1] = 1.0 - p_yxx[y][p][0];
				} else {
					p_yxx[y][p][0] -= delta;
					p_yxx[y][p][1] = 1.0 - p_yxx[y][p][0];
				}
			}
		}

	}

	private void intoducePosteriorDrift8(double p_yxx[][][], double delta) {

		//System.out.println("Before: " + p_yxx[0][0][0] + ", " + p_yxx[1][0][0] + ", " + p_yxx[0][0][1] + ", " + p_yxx[1][0][1] + ", " + p_yxx[0][1][0] + ", " + p_yxx[1][1][0] + ", " + p_yxx[0][1][1] + ", " + p_yxx[1][1][1] );
		int p;  // the parent
		int y;  // the class

		for (p = 0; p <= 1; p++) {
			for (y = 0; y <= 1; y++) {
				if (p_yxx[y][p][0] < delta)  {
					p_yxx[y][p][0] += delta;
					p_yxx[y][p][1] = 1.0 - p_yxx[y][p][0];
				} else if (p_yxx[y][p][0] > 1.0 - delta)  {
					p_yxx[y][p][0] -= delta;
					p_yxx[y][p][1] = 1.0 - p_yxx[y][p][0];
				} else if (r.nextInt(0, 1) == 0)  {
					p_yxx[y][p][0] += delta;
					p_yxx[y][p][1] = 1.0 - p_yxx[y][p][0];
				} else {
					p_yxx[y][p][0] -= delta;
					p_yxx[y][p][1] = 1.0 - p_yxx[y][p][0];
				}
			}
		}

		//System.out.println("Before: " + p_yxx[0][0][0] + ", " + p_yxx[1][0][0] + ", " + p_yxx[0][0][1] + ", " + p_yxx[1][0][1] + ", " + p_yxx[0][1][0] + ", " + p_yxx[1][1][0] + ", " + p_yxx[0][1][1] + ", " + p_yxx[1][1][1] );
	}

	private void intoducePosteriorDrift8(double p_yx[][], double delta) {

		//System.out.println("Before: " + p_yxx[0][0][0] + ", " + p_yxx[1][0][0] + ", " + p_yxx[0][0][1] + ", " + p_yxx[1][0][1] + ", " + p_yxx[0][1][0] + ", " + p_yxx[1][1][0] + ", " + p_yxx[0][1][1] + ", " + p_yxx[1][1][1] );
		int y;  // the class

		for (y = 0; y <= 1; y++) {
			if (p_yx[y][0] < delta)  {
				p_yx[y][0] += delta;
				p_yx[y][1] = 1.0 - p_yx[y][0];
			} else if (p_yx[y][0] > 1.0 - delta)  {
				p_yx[y][0] -= delta;
				p_yx[y][1] = 1.0 - p_yx[y][0];
			} else if (r.nextInt(0, 1) == 0)  {
				p_yx[y][0] += delta;
				p_yx[y][1] = 1.0 - p_yx[y][0];
			} else {
				p_yx[y][0] -= delta;
				p_yx[y][1] = 1.0 - p_yx[y][0];
			}
		}

		//System.out.println("Before: " + p_yxx[0][0][0] + ", " + p_yxx[1][0][0] + ", " + p_yxx[0][0][1] + ", " + p_yxx[1][0][1] + ", " + p_yxx[0][1][0] + ", " + p_yxx[1][1][0] + ", " + p_yxx[0][1][1] + ", " + p_yxx[1][1][1] );
	}

	// Assumption - all p value are initialised from the range (delta, 1.0-delta) and delta < 0.5
	private void intoducePosteriorDrift7(double[][][] p_yxx, double delta) {

		//System.out.println("Before: " + p_yxx[0][0][0] + ", " + p_yxx[1][0][0] + ", " + p_yxx[0][0][1] + ", " + p_yxx[1][0][1] + ", " + p_yxx[0][1][0] + ", " + p_yxx[1][1][0] + ", " + p_yxx[0][1][1] + ", " + p_yxx[1][1][1] );

		// first do parent = 0
		if (p_yxx[0][0][0] < delta)  {

			assert(p_yxx[0][0][1] >= delta);

			p_yxx[0][0][0] += delta;
			p_yxx[1][0][0] = 1.0 - p_yxx[0][0][0];

			p_yxx[0][0][1] -= delta;
			p_yxx[1][0][1] = 1.0 - p_yxx[0][0][1];

		} else if (p_yxx[0][0][0] > 1.0 - delta)  {

			assert(p_yxx[0][0][1] <= 1.0 - delta);

			p_yxx[0][0][0] -= delta;
			p_yxx[1][0][0] = 1.0 - p_yxx[0][0][0];

			p_yxx[0][0][1] += delta;
			p_yxx[1][0][1] = 1.0 - p_yxx[0][0][1];

		} else if (r.nextInt(0, 1) == 0)  {

			assert(p_yxx[0][0][1] >= delta);

			p_yxx[0][0][0] += delta;
			p_yxx[1][0][0] = 1.0 - p_yxx[0][0][0];

			p_yxx[0][0][1] -= delta;
			p_yxx[1][0][1] = 1.0 - p_yxx[0][0][1];

		} else {

			assert(p_yxx[0][0][1] <= 1.0 - delta);

			p_yxx[0][0][0] -= delta;
			p_yxx[1][0][0] = 1.0 - p_yxx[0][0][0];

			p_yxx[0][0][1] += delta;
			p_yxx[1][0][1] = 1.0 - p_yxx[0][0][1];
		}

		// then do parent = 1
		if (p_yxx[0][1][0] < delta)  {

			assert(p_yxx[0][1][1] >= delta);

			p_yxx[0][1][0] += delta;
			p_yxx[1][1][0] = 1.0 - p_yxx[0][1][0];

			p_yxx[0][1][1] -= delta;
			p_yxx[1][1][1] = 1.0 - p_yxx[0][1][1];

		} else if (p_yxx[0][1][0] > 1.0 - delta)  {

			assert(p_yxx[0][1][1] <= 1.0 - delta);

			p_yxx[0][1][0] -= delta;
			p_yxx[1][1][0] = 1.0 - p_yxx[0][1][0];

			p_yxx[0][1][1] += delta;
			p_yxx[1][1][1] = 1.0 - p_yxx[0][1][1];

		} else if (r.nextInt(0, 1) == 0)  {

			assert(p_yxx[0][1][1] >= delta);

			p_yxx[0][1][0] += delta;
			p_yxx[1][1][0] = 1.0 - p_yxx[0][1][0];

			p_yxx[0][1][1] -= delta;
			p_yxx[1][1][1] = 1.0 - p_yxx[0][1][1];

		} else {

			assert(p_yxx[0][0][1] <= 1.0 - delta);

			p_yxx[0][1][0] -= delta;
			p_yxx[1][1][0] = 1.0 - p_yxx[0][1][0];

			p_yxx[0][1][1] += delta;
			p_yxx[1][1][1] = 1.0 - p_yxx[0][1][1];
		}

		//System.out.println("Before: " + p_yxx[0][0][0] + ", " + p_yxx[1][0][0] + ", " + p_yxx[0][0][1] + ", " + p_yxx[1][0][1] + ", " + p_yxx[0][1][0] + ", " + p_yxx[1][1][0] + ", " + p_yxx[0][1][1] + ", " + p_yxx[1][1][1] );
	}

	// Assumption - all p value are initialised from the range (delta, 1.0-delta) and delta < 0.5
	private void intoducePosteriorDrift7(double[][] p_yx, double delta) {

		//System.out.println("Before: " + p_yxx[0][0][0] + ", " + p_yxx[1][0][0] + ", " + p_yxx[0][0][1] + ", " + p_yxx[1][0][1] + ", " + p_yxx[0][1][0] + ", " + p_yxx[1][1][0] + ", " + p_yxx[0][1][1] + ", " + p_yxx[1][1][1] );

		// first do parent = 0
		if (p_yx[0][0] < delta)  {

			assert(p_yx[0][1] >= delta);

			p_yx[0][0] += delta;
			p_yx[1][0] = 1.0 - p_yx[0][0];

			p_yx[0][1] -= delta;
			p_yx[1][1] = 1.0 - p_yx[0][1];

		} else if (p_yx[0][0] > 1.0 - delta)  {

			assert(p_yx[0][1] <= 1.0 - delta);

			p_yx[0][0] -= delta;
			p_yx[1][0] = 1.0 - p_yx[0][0];

			p_yx[0][1] += delta;
			p_yx[1][1] = 1.0 - p_yx[0][1];

		} else if (r.nextInt(0, 1) == 0)  {

			assert(p_yx[0][1] >= delta);

			p_yx[0][0] += delta;
			p_yx[1][0] = 1.0 - p_yx[0][0];

			p_yx[0][1] -= delta;
			p_yx[1][1] = 1.0 - p_yx[0][1];

		} else {

			assert(p_yx[0][1] <= 1.0 - delta);

			p_yx[0][0] -= delta;
			p_yx[1][0] = 1.0 - p_yx[0][0];

			p_yx[0][1] += delta;
			p_yx[1][1] = 1.0 - p_yx[0][1];
		}

		//System.out.println("Before: " + p_yxx[0][0][0] + ", " + p_yxx[1][0][0] + ", " + p_yxx[0][0][1] + ", " + p_yxx[1][0][1] + ", " + p_yxx[0][1][0] + ", " + p_yxx[1][1][0] + ", " + p_yxx[0][1][1] + ", " + p_yxx[1][1][1] );
	}

	private void intoducePosteriorDrift6(double[][][] p_yxx, double delta, double gamma, double sign) {

		//System.out.println("Before: " + p_yxx[0][0][0] + ", " + p_yxx[1][0][0] + ", " + p_yxx[0][0][1] + ", " + p_yxx[1][0][1] + ", " + p_yxx[0][1][0] + ", " + p_yxx[1][1][0] + ", " + p_yxx[0][1][1] + ", " + p_yxx[1][1][1] );

		// first do parent = 0
		if (r.nextInt(0, 1) <= 0.5)  {
			p_yxx[0][0][0] += delta;
			if (p_yxx[0][0][0] > 1.0) p_yxx[0][0][0] = 1.0;
			p_yxx[1][0][0] = 1.0 - p_yxx[0][0][0];

			p_yxx[0][0][1] -= delta;
			if (p_yxx[0][0][1] < 0.0) p_yxx[0][0][1] = 0.0;
			p_yxx[1][0][1] = 1.0 - p_yxx[0][0][1];
		}
		else {
			p_yxx[0][0][0] -= delta;
			if (p_yxx[0][0][0] < 0.0) p_yxx[0][0][0] = 0.0;
			p_yxx[1][0][0] = 1.0 - p_yxx[0][0][0];

			p_yxx[0][0][1] += delta;
			if (p_yxx[0][0][1] > 1.0) p_yxx[0][0][1] = 1.0;
			p_yxx[1][0][1] = 1.0 - p_yxx[0][0][1];
		}

		// then do parent = 1
		if (r.nextInt(0, 1) <= 0.5)  {
			p_yxx[0][1][0] += delta;
			if (p_yxx[0][1][0] > 1.0) p_yxx[0][1][0] = 1.0;
			p_yxx[1][1][0] = 1.0 - p_yxx[0][1][0];

			p_yxx[0][1][1] -= delta;
			if (p_yxx[0][1][1] < 0.0) p_yxx[0][1][1] = 0.0;
			p_yxx[1][1][1] = 1.0 - p_yxx[0][1][1];
		}
		else {
			p_yxx[0][1][0] -= delta;
			if (p_yxx[0][1][0] < 0.0) p_yxx[0][1][0] = 0.0;
			p_yxx[1][1][0] = 1.0 - p_yxx[0][1][0];

			p_yxx[0][1][1] += delta;
			if (p_yxx[0][1][1] > 1.0) p_yxx[0][1][1] = 1.0;
			p_yxx[1][1][1] = 1.0 - p_yxx[0][1][1];
		}

		//System.out.println("Before: " + p_yxx[0][0][0] + ", " + p_yxx[1][0][0] + ", " + p_yxx[0][0][1] + ", " + p_yxx[1][0][1] + ", " + p_yxx[0][1][0] + ", " + p_yxx[1][1][0] + ", " + p_yxx[0][1][1] + ", " + p_yxx[1][1][1] );
	}

	private void intoducePosteriorDrift5(double[][][] p_yxx, double delta, double gamma, double sign) {

		//System.out.println("Before: " + p_yxx[0][0][0] + ", " + p_yxx[1][0][0] + ", " + p_yxx[0][0][1] + ", " + p_yxx[1][0][1] + ", " + p_yxx[0][1][0] + ", " + p_yxx[1][1][0] + ", " + p_yxx[0][1][1] + ", " + p_yxx[1][1][1] );

		if (sign == 1) {
			if (p_yxx[0][0][0] <= (delta)) 		p_yxx[0][0][0] += delta;
			if (p_yxx[1][0][0] >= (1 - delta)) p_yxx[1][0][0] -= delta;
			if (p_yxx[0][0][1] >= (1 - delta)) p_yxx[0][0][1] -= delta;
			if (p_yxx[1][0][1] <= (delta)) 		p_yxx[1][0][1] += delta;

			if (p_yxx[0][1][0] <= (delta)) 		p_yxx[0][1][0] += gamma;
			if (p_yxx[1][1][0] >= (1 - delta)) p_yxx[1][1][0] -= gamma;
			if (p_yxx[0][1][1] >= (1 - delta)) p_yxx[0][1][1] -= gamma;
			if (p_yxx[1][1][1] <= (delta)) 		p_yxx[1][1][1] += gamma;

		} else if (sign == -1) {
			if (p_yxx[0][0][0] >= (1 - delta)) 		p_yxx[0][0][0] -= delta;
			if (p_yxx[1][0][0] <= (delta)) 				p_yxx[1][0][0] += delta;
			if (p_yxx[0][0][1] <= (delta)) 				p_yxx[0][0][1] += delta;
			if (p_yxx[1][0][1] >= (1 - delta)) 		p_yxx[1][0][1] -= delta;

			if (p_yxx[0][1][0] >= (1 - delta)) 		p_yxx[0][1][0] -= gamma;
			if (p_yxx[1][1][0] <= (delta)) 				p_yxx[1][1][0] += gamma;
			if (p_yxx[0][1][1] <= (delta)) 				p_yxx[0][1][1] += gamma;
			if (p_yxx[1][1][1] >= (1 - delta)) 		p_yxx[1][1][1] -= gamma;
		}

		//System.out.println("Before: " + p_yxx[0][0][0] + ", " + p_yxx[1][0][0] + ", " + p_yxx[0][0][1] + ", " + p_yxx[1][0][1] + ", " + p_yxx[0][1][0] + ", " + p_yxx[1][1][0] + ", " + p_yxx[0][1][1] + ", " + p_yxx[1][1][1] );
	}

	private void intoducePosteriorDrift4(double[][][] p_yxx, double delta, double gamma, double sign) {

		//System.out.println("Before: " + p_yxx[0][0][0] + ", " + p_yxx[1][0][0] + ", " + p_yxx[0][0][1] + ", " + p_yxx[1][0][1] + ", " + p_yxx[0][1][0] + ", " + p_yxx[1][1][0] + ", " + p_yxx[0][1][1] + ", " + p_yxx[1][1][1] );

		double newval = p_yxx[0][0][0] + (sign * delta);
		if (newval > 1) {
			p_yxx[0][0][0] = 1;
		} else if (newval < 0) {
			p_yxx[0][0][0] = 0;
		}

		newval = p_yxx[1][0][0] - (sign * delta);
		if (newval > 1) {
			p_yxx[1][0][0] = 1;
		} else if (newval < 0) {
			p_yxx[1][0][0] = 0;
		}

		newval = p_yxx[0][0][1] - (sign * delta);
		if (newval > 1) {
			p_yxx[0][0][1] = 1;
		} else if (newval < 0) {
			p_yxx[0][0][1] = 0;
		}

		newval = p_yxx[1][0][1] + (sign * delta);
		if (newval > 1) {
			p_yxx[1][0][1] = 1;
		} else if (newval < 0) {
			p_yxx[1][0][1] = 0;
		}

		newval = p_yxx[0][1][0] + (sign * gamma);
		if (newval > 1) {
			p_yxx[0][1][0] = 1;
		} else if (newval < 0) {
			p_yxx[0][1][0] = 0;
		}

		newval = p_yxx[1][1][0] - (sign * gamma);
		if (newval > 1) {
			p_yxx[1][1][0] = 1;
		} else if (newval < 0) {
			p_yxx[1][1][0] = 0;
		}

		newval = p_yxx[0][1][1] - (sign * gamma);
		if (newval > 1) {
			p_yxx[0][1][1] = 1;
		} else if (newval < 0) {
			p_yxx[0][1][1] = 0;
		}

		newval = p_yxx[1][1][1] + (sign * gamma);
		if (newval > 1) {
			p_yxx[1][1][1] = 1;
		} else if (newval < 0) {
			p_yxx[1][1][1] = 0;
		}

		//System.out.println("Before: " + p_yxx[0][0][0] + ", " + p_yxx[1][0][0] + ", " + p_yxx[0][0][1] + ", " + p_yxx[1][0][1] + ", " + p_yxx[0][1][0] + ", " + p_yxx[1][1][0] + ", " + p_yxx[0][1][1] + ", " + p_yxx[1][1][1] );
	}

	private void intoducePosteriorDrift2(double[][][] p_yxx, double delta, double gamma, double sign) {

		int nvals = 2;
		int nc = 2;

		for (int y = 0; y < nc; y++) {
			for (int x1 = 0; x1 < nvals; x1++) {

				double sum = 0;
				for (int x2 = 0; x2 < nvals; x2++) {
					p_yxx[y][x1][x2] = r.nextUniform(0, 1);

					sum += p_yxx[y][x1][x2];
				}

				for (int x2 = 0; x2 < nvals; x2++) {
					p_yxx[y][x1][x2] /= sum;
				}

			}
		}
	}

	private void intoducePosteriorDrift2(double[][] p_yx, double delta, double gamma, double sign) {

		int nvals = 2;
		int nc = 2;

		for (int y = 0; y < nc; y++) {

			double sum = 0;
			for (int x1 = 0; x1 < nvals; x1++) {
				p_yx[y][x1] = r.nextUniform(0, 1);

				sum += p_yx[y][x1];
			}

			for (int x1 = 0; x1 < nvals; x1++) {
				p_yx[y][x1] /= sum;
			}

		}
	}

	private void intoducePosteriorDrift3(double[][][] p_yxx, double delta, double gamma, double sign) {

		p_yxx[0][0][0] += (sign * delta);
		p_yxx[1][0][0] -= (sign * delta);
		p_yxx[0][0][1] -= (sign * delta);
		p_yxx[1][0][1] += (sign * delta);

		p_yxx[0][1][0] += (sign * gamma);
		p_yxx[1][1][0] -= (sign * gamma);
		p_yxx[0][1][1] -= (sign * gamma);
		p_yxx[1][1][1] += (sign * gamma);
	}

	private void intoducePosteriorDrift3(double[][] p_yx, double delta, double sign) {

		p_yx[0][0] += (sign * delta);
		p_yx[1][0] -= (sign * delta);		
		p_yx[0][1] -= (sign * delta);
		p_yx[1][1] += (sign * delta);
	}

	@Override
	protected void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {

		RandomGenerator rg = new JDKRandomGenerator();
		rg.setSeed(seed.getValue());
		r = new RandomDataGenerator(rg);

		generateHeader();

		int nc = Globals.getNumClasses();
		int n = streamHeader.numAttributes() - 1;
		int nvals = 2;

		parents = new int[n];
		parents[1] = 0;
		for (int i = 2; i < n; i++) {
			//int p = r.nextInt(0, i-1);
			int p  = 0;
			parents[i] = p;
		}
		System.out.println("Seed = " + seed.getValue());
		System.out.println(Arrays.toString(parents));

		/* Declaration */

		p_y = new double[nc];

		p_yx = new double[n][][];

		p_yx[0] = new double[nc][];
		for (int y = 0; y < nc; y++) {
			p_yx[0][y] = new double[nvals];
		}

		for (int i = 1; i < m_Q; i++) {
			p_yx[i] = new double[nc][];
			for (int y = 0; y < nc; y++) {
				p_yx[i][y] = new double[nvals];
			}
		}

		p_yxx = new double[n][][][];

		for (int i = m_Q; i < n; i++) {
			p_yxx[i] = new double[nc][][];
			for (int y = 0; y < nc; y++) {
				p_yxx[i][y] = new double[nvals][];
				for (int x = 0; x < nvals; x++) {
					p_yxx[i][y][x] = new double[nvals];
				}
			}
		}

		d_yx = new int[n][];

		for (int i = 0; i < m_Q; i++) {
			d_yx[i] = new int[nc];
		}

		d_yxx = new int[n][][];

		for (int i = m_Q; i < n; i++) {
			d_yxx[i] = new int[nc][];
			for (int y = 0; y < nc; y++) {
				d_yxx[i][y] = new int[nvals];
			}
		}

		/* Initialization */
		intializeCPTBinary();

		nInstancesGeneratedSoFar = 0L;
	}

	public void intializeCPTBinary() {

		int nc = Globals.getNumClasses();
		int n = streamHeader.numAttributes() - 1;
		int nvals = 2;

		p_y[0] = r.nextUniform(0, 1);
		p_y[1] = 1.0 - p_y[0];

		for (int y = 0; y < nc; y++) {
			p_yx[0][y][0] = r.nextUniform(0, 1);
			p_yx[0][y][1] = 1.0 - p_yx[0][y][0];
		}

		for (int i = 1; i < m_Q; i++) {
			for (int y = 0; y < nc; y++) {
				p_yx[i][y][0] = r.nextUniform(0, 1);
				p_yx[i][y][1] = 1.0 - p_yx[i][y][0];
			}
		}

		for (int i = m_Q; i < n; i++) {
			for (int y = 0; y < nc; y++) {
				for (int x1 = 0; x1 < nvals; x1++) {
					p_yxx[i][y][x1][0] = r.nextUniform(0, 1);
					p_yxx[i][y][x1][1] = 1 - p_yxx[i][y][x1][0];
				}
			}
		}

		for (int i = 0; i < m_Q; i++) {
			for (int y = 0; y < nc; y++) {
				d_yx[i][y] = r.nextInt(0, 1);
				d_yx[i][y] = r.nextInt(0, 1);
			}
		}

		for (int i = m_Q; i < n; i++) {
			for (int y = 0; y < nc; y++) {
				for (int x1 = 0; x1 < nvals; x1++) {
					d_yxx[i][y][x1] = r.nextInt(0, 1);
					d_yxx[i][y][x1] = r.nextInt(0, 1);
				}
			}
		}

	}

}
