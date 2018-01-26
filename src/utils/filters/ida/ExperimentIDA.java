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

import java.text.DecimalFormat;
import java.util.ArrayList;

import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.functions.SGDMultiClass;
import moa.core.TimingUtils;
import moa.streams.ArffFileStream;

import utils.filters.ida.IDADiscretizer.IDAType;

public class ExperimentIDA {
	
	static DecimalFormat df = new DecimalFormat("#.####");
	
	public static void main(String[] args) throws Exception {
//		String[] datasets = new String[]{"airlines-n.arff","elecNormNew-n.arff","gas-sensor-n.arff","powersupply-n.arff","sensor-n.arff"};
		String[] datasets = new String[]{"airlines-n.arff","elecNormNew-n.arff","gas-sensor-n.arff","powersupply-n.arff","sensor-n.arff"};
		
		for (String string : datasets) {
			System.out.println(string.substring(0,string.lastIndexOf("."))+"\t"+(df.format(classifyOriginal(string)))+" "+(df.format(classifyDiscretized(string))));
		}
	}
	
	public static double classifyDiscretized(String dataset) {
		SGDMultiClass learner = new SGDMultiClass();
		ArffFileStream stream = new ArffFileStream("/home/forestier/Dropbox/#decay/datasets/clean/"+dataset, -1);
        stream.prepareForUse();
        
        IDADiscretizer filter = new IDADiscretizer(5,1000,IDAType.IDAW);
        filter.setInputStream(stream);
        filter.init();
        filter.prepareForUse();
        
		int numInstances = Integer.MAX_VALUE;

        learner.setModelContext(filter.getHeader());
        learner.prepareForUse();
		learner.setLossFunction(1);
		learner.setLambda(0.001);
		learner.setLearningRate(0.01);
        
        int numberSamplesCorrect = 0;
        int numberSamples = 0;
        long evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
        while (filter.hasMoreInstances() && numberSamples < numInstances) {
                Instance trainInst = filter.nextInstance().getData();
                
                boolean predict = learner.correctlyClassifies(trainInst);
                if (predict){
                        numberSamplesCorrect++;
                }
                numberSamples++;
                
                learner.trainOnInstance(trainInst);
        }
        double accuracy = 1 -( (double) numberSamplesCorrect/ (double) numberSamples);
//        double time = TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread()- evaluateStartTime);
//        System.out.println(numberSamples + " instances processed with " + accuracy + " error rate in "+time+" seconds.");
        return accuracy;
	}
	
	public static double classifyOriginal(String dataset) {
		SGDMultiClass learner = new SGDMultiClass();
		ArffFileStream filter = new ArffFileStream("/home/forestier/Dropbox/#decay/datasets/clean/"+dataset, -1);
        filter.prepareForUse();
        
		int numInstances = Integer.MAX_VALUE;

        learner.setModelContext(filter.getHeader());
        learner.prepareForUse();
		learner.setLossFunction(1);
		learner.setLambda(0.001);
		learner.setLearningRate(0.01);

        int numberSamplesCorrect = 0;
        int numberSamples = 0;
        long evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
        while (filter.hasMoreInstances() && numberSamples < numInstances) {
                Instance trainInst = filter.nextInstance().getData();
                
                boolean predict = learner.correctlyClassifies(trainInst);
                if (predict){
                        numberSamplesCorrect++;
                }
                numberSamples++;
                
                learner.trainOnInstance(trainInst);
        }
        double accuracy = 1 -( (double) numberSamplesCorrect/ (double) numberSamples);
        double time = TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread()- evaluateStartTime);
//        System.out.println(numberSamples + " instances processed with " + accuracy + " error rate in "+time+" seconds.");
        return accuracy;
	}

	public static void main1(String[] args) throws Exception {
		SGDMultiClass learner = new SGDMultiClass();
		learner.setLossFunction(1);
		learner.setLambda(0.001);
		learner.setLearningRate(0.01);
//		NaiveBayes learner = new NaiveBayes();
		
//		SGD test = new SGD();

		ArffFileStream stream = new ArffFileStream("/home/forestier/Dropbox/#decay/datasets/elecNormNew.arff", -1);
        stream.prepareForUse();
        
        IDADiscretizer filter = new IDADiscretizer(5,1000,IDAType.IDAW);
        filter.setInputStream(stream);
        filter.init();
        filter.prepareForUse();
        
		int numInstances = Integer.MAX_VALUE;
		boolean isTesting = true;

        learner.setModelContext(filter.getHeader());
        learner.prepareForUse();
        
        ArrayList<Boolean> predictions = new ArrayList<Boolean>();

        int numberSamplesCorrect = 0;
        int numberSamples = 0;
        long evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
        while (filter.hasMoreInstances() && numberSamples < numInstances) {
                Instance trainInst = filter.nextInstance().getData();
//                System.out.println(trainInst);
                if (isTesting) {
//                	System.out.println(Arrays.toString(learner.getVotesForInstance(trainInst)));
                	boolean predict = learner.correctlyClassifies(trainInst);
                    if (predict){
                            numberSamplesCorrect++;
                    }
                    predictions.add(predict);
                }
                numberSamples++;
                
                learner.trainOnInstance(trainInst);
                if(numberSamples > 10) {
                	int count = 0;
                	for (int i = 1; i <= 10; i++) {
						if(predictions.get(numberSamples - i))
							count++;
					}
//                	System.out.println(numberSamples+" "+count/10.0);
                }
        }
        double accuracy = 1 -( (double) numberSamplesCorrect/ (double) numberSamples);
        double time = TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread()- evaluateStartTime);
        System.out.println(numberSamples + " instances processed with " + accuracy + " error rate in "+time+" seconds.");
	}
}
