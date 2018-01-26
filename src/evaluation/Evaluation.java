package evaluation;

import utils.InputArguments;
import utils.SanctityCheck;

import java.io.File;

import utils.Globals;

public class Evaluation {

	public static void main(String[] args) throws Exception {
		
		InputArguments options = new InputArguments();
		
		options.setOptions(args);

		if (!SanctityCheck.checkStringArgs()) {
			System.out.print("Please correct your input arguments, ... Exiting()");
			System.exit(-1);
		} else {
			SanctityCheck.printExperimentInformation();
		}

		String val = Globals.getExperimentType();

		if (val.equalsIgnoreCase("traintest")) {

			String ds = SanctityCheck.determineDSValue();
			Globals.setDataSetName(ds);

			evaluationTrainTest.learn();

		} else if (val.equalsIgnoreCase("prequential")) {

			String ds = SanctityCheck.determineDSValue();
			Globals.setDataSetName(ds);

			evaluationPrequential.learn();

		} else if (val.equalsIgnoreCase("flowMachines")) {

			evaluationFlowMachines.learn();

		}  else if (val.equalsIgnoreCase("drift")) {

			evaluationDrift.learn();

		} 

	}

}
