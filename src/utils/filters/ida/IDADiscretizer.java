/* 
** Class for a discretisation filter for instance streams
** Copyright (C) 2016 Germain Forestier, Geoffrey I Webb
**
** This program is free software: you can redistribute it and/or modify
** it under the terms of the GNU General Public License as published by
** the Free Software Foundation, either version 3 of the License, or
** (at your option) any later version.
** 
** This program is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
** GNU General Public License for more details.
** 
** You should have received a copy of the GNU General Public License
** along with this program. If not, see <http://www.gnu.org/licenses/>.
**
** Please report any bugs to Germain Forestier <germain.forestier@uha.fr>
*/
package utils.filters.ida;

import java.util.Random;

import org.apache.commons.lang3.NotImplementedException;

import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;

import moa.core.FastVector;
import moa.core.InstanceExample;
import moa.streams.InstanceStream;
import moa.streams.filters.AbstractStreamFilter;

public class IDADiscretizer extends AbstractStreamFilter {
	private static final long serialVersionUID = 1L;
	
	// number of bins for each numerical attributes
	protected int nBins;
	// number of samples (i.e. size of the window)
	protected int sampleSize;
	// new header with discretized attributes
	protected InstancesHeader discretizedHeader;
	// number of instances seen so far
	protected int nbSeenInstances;
	// number of attributes
	protected int nbAttributes;
	// number of attributes
	protected int nbNumericalAttributes;
	// sample reservoir, one for each numerical attribute
	protected SamplingReservoir[] sReservoirs;
	
	// type of IDA
	protected IDAType type;
	public enum IDAType {
		IDA,
		IDAW
	}
	
	// has been init
	protected boolean init = false;

	/**
	 * Create an IDA filter
	 * @param nBins number of bins
	 * @param sampleSize number of samples
	 * @param window or random
	 */
	public IDADiscretizer(int nBins, int sampleSize, IDAType type) {
		this.nBins = nBins;
		this.sampleSize = sampleSize;
		this.type = type;
	}

	@Override
	public InstancesHeader getHeader() {
		return this.discretizedHeader;
	}

	@Override
	public void getDescription(StringBuilder sb, int indent) {
		throw new NotImplementedException("Not implemented");
	}

	@Override
	protected void restartImpl() {
		this.nBins = nBins;
		this.sampleSize = sampleSize;
		this.type = type;
		this.init = false;
	}

	@Override
	public InstanceExample nextInstance() {
		if(!init) {
			init();
		}
		// one more seen instance
		nbSeenInstances++;
		
		InstanceExample instEx = (InstanceExample) this.inputStream.nextInstance().copy();
		Instance inst = instEx.getData();

		Instance discretizedInstance = new DenseInstance(discretizedHeader.numAttributes());
		discretizedInstance.setDataset(discretizedHeader);
		
		if(type.equals(IDAType.IDA)) { // random sample
			updateRandomSample(inst); 
		} else if(type.equals(IDAType.IDAW)) { // window sample
			updateWindowSample(inst);
		}
		
		int nbNumericalAttributesCount = 0;
		 for (int i = 0; i < this.nbAttributes; i++) {
			 // if numeric and not missing, discretize
			 if(inst.attribute(i).isNumeric() && !inst.isMissing(i)) {
				 double v = inst.value(i);
				 int bin = sReservoirs[nbNumericalAttributesCount].getBin(v);
				 discretizedInstance.setValue(i,bin);
				 nbNumericalAttributesCount++;
			 } else {
				 if(!inst.isMissing(i)) {
					 discretizedInstance.setValue(i,inst.value(i));
				 } else {
					 discretizedInstance.setValue(i,Double.NaN);
				 }
			 }
		 }
		 // set the class of the instance
		 discretizedInstance.setClassValue(inst.classValue());
		 
		 return new InstanceExample(discretizedInstance);
	}
	
	/**
	 * Window sample (IDAW)
	 * @param inst the new instance
	 */
	protected void updateWindowSample(Instance inst) {
		int nbNumericalAttributesCount = 0;
		for (int i = 0; i < this.nbAttributes; i++) {
			double v = inst.value(i);
			// if the value is not missing, then add it to the pool
			if(inst.attribute(i).isNumeric() && !inst.isMissing(i)) {
				this.sReservoirs[nbNumericalAttributesCount].insertWithWindow(v);
				if(!this.sReservoirs[nbNumericalAttributesCount].checkValueInQueues(v)) {
					System.err.println("Value not added.");
				}
			}
			if(inst.attribute(i).isNumeric()) { 
				nbNumericalAttributesCount++;
			}
		}
	}
	
	/**
	 * Random sample (IDA)
	 * @param inst the new instance
	 */
	protected void updateRandomSample(Instance inst) {
		int nbNumericalAttributesCount = 0;
		for (int i = 0; i < this.nbAttributes; i++) {
			double v = inst.value(i);
			// if the value is not missing, then add it to the pool
			if(inst.attribute(i).isNumeric() && !inst.isMissing(i)) {
				if(sReservoirs[nbNumericalAttributesCount].getNbSamples() < sampleSize) {
					this.sReservoirs[nbNumericalAttributesCount].insertValue(v);
				} else {
					double rValue = Math.random();
					if(rValue <= (double)sampleSize/(double)nbSeenInstances) {
						int randval = new Random().nextInt(sampleSize);
						this.sReservoirs[nbNumericalAttributesCount].replace(randval,v);
					}
				}
			}
			if(inst.attribute(i).isNumeric()) { 
				nbNumericalAttributesCount++;
			}
		}
	}
	
	/**
	 * Init the stream
	 */
	public void init() {
		generateNewHeader();
		this.init = true;
		// minus one for the class
		this.nbAttributes = this.getHeader().numAttributes() - 1;
		this.sReservoirs =  new SamplingReservoir[nbNumericalAttributes];
		for (int i = 0; i < nbNumericalAttributes; i++) {
			sReservoirs[i] = new SamplingReservoir(this.nBins, this.sampleSize, i);
		}
	}

	/**
	 * Create the header for the discretized instance
	 */
	protected void generateNewHeader() {
		InstancesHeader streamHeader = this.inputStream.getHeader();
//		System.out.println(this.inputStream.getHeader());
		int nbAttributes = streamHeader.numAttributes();
		FastVector attributes = new FastVector();
		for (int i = 0; i < nbAttributes; i++) {
			Attribute attr = streamHeader.attribute(i);
			// create a new categorial attribute
			if (attr.isNumeric()) {
				FastVector newAttrLabels = new FastVector();
				for (int j = 0; j < nBins; j++) {
					newAttrLabels.add("b"+j);
				}

				attributes.addElement(new Attribute(attr.name(), newAttrLabels));
				nbNumericalAttributes++;

			} else {
				attributes.addElement(attr);
			}
		}
		discretizedHeader = new InstancesHeader(new Instances(getCLICreationString(InstanceStream.class), attributes, 0));
//		discretizedHeader.setClassIndex(this.getHeader().classIndex());
		// TODO better handle class attribute
		discretizedHeader.setClassIndex(discretizedHeader.numAttributes()-1);
	}
}
