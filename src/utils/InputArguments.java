package utils;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

public class InputArguments {

	private Options options = null;

	public InputArguments() {

		Option help = new Option( "help", "print this message" );
		Option version = new Option( "version", "print the version information and exit" );
		Option verbose = new Option( "verbose", "be extra verbose" );

		Option experimentType   = OptionBuilder.withArgName( "experimentType" ).hasArg().withDescription("Experiment Type: <cv, traintest, prequential, recommender>" ).create( "experimentType" );

		Option trainFile   = OptionBuilder.withArgName( "trainFile" ).hasArg().withDescription("Specify valid ARFF File for Training" ).create( "trainFile" );
		Option testFile   = OptionBuilder.withArgName( "testFile" ).hasArg().withDescription("Specify valid ARFF File for Testing" ).create( "testFile" );
		Option cvFile   = OptionBuilder.withArgName( "cvFile" ).hasArg().withDescription("Specify valid ARFF File for Testing" ).create( "cvFile" );

		Option arffBufferSize   = OptionBuilder.withArgName( "arffBufferSize" ).hasArg().withDescription("Arff Buffer Size" ).create( "arffBufferSize" );
		Option bufferSize   = OptionBuilder.withArgName( "bufferSize" ).hasArg().withDescription("Buffer Size" ).create( "bufferSize" );

		Option numExp   = OptionBuilder.withArgName( "numExp" ).hasArg().withDescription("Number of Experiments to Run" ).create( "numExp" );
		Option numFolds   = OptionBuilder.withArgName( "numFolds" ).hasArg().withDescription("Number of Folds to train in case of cv Experiments" ).create( "numFolds" );

		Option discretization   = OptionBuilder.withArgName( "discretization" ).hasArg().withDescription("Discretize Data: <none, mdl, ef, ew>" ).create( "discretization" );
		Option discretizationParameter   = OptionBuilder.withArgName( "discretizationParameter" ).hasArg().withDescription("Input parameter for Discretization" ).create( "discretizationParameter" );
		Option normalizeNumeric = new Option( "normalizeNumeric", "Normalize Numeric Attributes" );
		Option holdoutsetPrecentage   = OptionBuilder.withArgName( "holdoutsetPrecentage" ).hasArg().withDescription("Percentage of Hold-out data, default is 5%" ).create( "holdoutsetPrecentage" );

		Option model   = OptionBuilder.withArgName( "model" ).hasArg().withDescription("Model: <ALR, AnJE, KDB, FM, ANN, AnDE>" ).create( "model" );
		Option level   = OptionBuilder.withArgName( "level" ).hasArg().withDescription("Level of Model" ).create( "level" );
		Option doWanbiac   = new Option( "doWanbiac", "Use WANBIA-C trick" );
		Option doDiscriminative  = new Option( "doDiscriminative", "Do Discrimnative Learning (SGD or MCMC)" );
		Option doSelectiveKDB  = new Option( "doSelectiveKDB", "Do Selective KDB" );
		Option doRegularization = new Option( "doRegularization", "Do Regularization" );
		Option lambda   = OptionBuilder.withArgName( "lambda" ).hasArg().withDescription("Lambda value for regularization" ).create( "lambda" );
		Option adaptiveRegularization   = OptionBuilder.withArgName( "adaptiveRegularization" ).hasArg().withDescription("Adaptive Regularization: <None, Rendle, Provost, Zaidi1, Zaidi2>" ).create( "adaptiveRegularization" );
		Option regularizationType   = OptionBuilder.withArgName( "regularizationType" ).hasArg().withDescription("Regularization type: <L2>" ).create( "regularizationType" );
		Option regularizationTowards   = OptionBuilder.withArgName( "regularizationTowards" ).hasArg().withDescription("Regularization towards a value" ).create( "regularizationTowards" );

		Option optimizer   = OptionBuilder.withArgName( "optimizer" ).hasArg().withDescription("Model: <SGD, MCMC>" ).create( "optimizer" );
		Option numIterations   = OptionBuilder.withArgName( "numIterations" ).hasArg().withDescription("No. of Iterations" ).create( "numIterations" );
		Option sgdType   = OptionBuilder.withArgName( "sgdType" ).hasArg().withDescription("SGD type: <plainsgd, adagrad, adadelta, nplr>" ).create( "sgdType" );
		Option sgdTuningParameter   = OptionBuilder.withArgName( "sgdTuningParameter" ).hasArg().withDescription("Specify ONE Tuning Parameter for SGD" ).create( "sgdTuningParameter" );		
		Option doCrossValidateTuningParameter   = new Option( "doCrossValidateTuningParameter", "Cross-validate SGD tuning Parameter" );

		Option mcmcType   = OptionBuilder.withArgName( "mcmcType" ).hasArg().withDescription("MCMC type: <ALS>" ).create( "mcmcType" );
		Option featureSelection   = OptionBuilder.withArgName( "featureSelection" ).hasArg().withDescription("Feature Selection: <None, MI, Count, ChiSqTest, GTest, FisherExactTest, AnJETest, AnJETestLOOCV, ALRTest>" ).create( "featureSelection" );
		Option featureSelectionParameter   = OptionBuilder.withArgName( "featureSelectionParameter" ).hasArg().withDescription("Specify ONE Parameter for Feature Selection" ).create( "featureSelectionParameter" );		

		Option objectiveFunction   = OptionBuilder.withArgName( "objectiveFunction" ).hasArg().withDescription("Objective Function: <CLL, MSE, HL>" ).create( "objectiveFunction" );
		Option doBinaryClassification = new Option( "doBinaryClassification", "Do 1vsAll Binary Classification" );
		Option dataStructureParameter   = OptionBuilder.withArgName( "dataStructureParameter" ).hasArg().withDescription("Parameter Structure: <Flat, IndexedBig, BitMap, Hash>" ).create( "dataStructureParameter" );
		Option hashParameter   = OptionBuilder.withArgName( "hashParameter" ).hasArg().withDescription("Hash Parameter (length of hashed vector)" ).create( "hashParameter" );

		Option adaptiveControl   = OptionBuilder.withArgName( "adaptiveControl" ).hasArg().withDescription("Adaptive Control: <None, Decay, Window>" ).create( "adaptiveControl" );
		Option adaptiveControlParameter   = OptionBuilder.withArgName( "adaptiveControlParameter" ).hasArg().withDescription("Parameter of Adaptive Control" ).create( "adaptiveControlParameter" );

		Option prequentialOutputResolution   = OptionBuilder.withArgName( "prequentialOutputResolution" ).hasArg().withDescription("Parameter Controlling output resolution of Prequential Plots" ).create( "prequentialOutputResolution" );

		Option prequentialBufferOutputResolution   = OptionBuilder.withArgName( "prequentialBufferOutputResolution" ).hasArg().withDescription("Parameter Controlling output buffer resolution of Prequential Plots" ).create( "prequentialBufferOutputResolution" );
		Option doMovingAverage = new Option( "doMovingAverage", "Plot Prequential Curves with Moving averages" );

		Option tempDirectory   = OptionBuilder.withArgName( "tempDirectory" ).hasArg().withDescription("Directory to store tmp files. Default: /tmp/" ).create( "tempDirectory" );
		Option ouputResultsDirectory   = OptionBuilder.withArgName( "ouputResultsDirectory" ).hasArg().withDescription("Directory to store Result files. Default: /tmp/" ).create( "ouputResultsDirectory" );

		Option driftType   = OptionBuilder.withArgName( "driftType" ).hasArg().withDescription("Drift Type. E.g., None, Bayesian, LR, Abrupt" ).create( "driftType" );

		Option flowParameter   = OptionBuilder.withArgName( "flowParameter" ).hasArg().withDescription("Flow Parameters. E.g. adaptiveControl:Decay: {0.2,0.4} or adaptiveControl:Window: {10,20}" ).create( "flowParameter" );

		Option driftMagnitude   = OptionBuilder.withArgName( "driftMagnitude" ).hasArg().withDescription("Drfit Magnitude" ).create( "driftMagnitude" );
		Option driftMagnitude2   = OptionBuilder.withArgName( "driftMagnitude2" ).hasArg().withDescription("Drfit Magnitude2" ).create( "driftMagnitude2" );
		Option driftMagnitude3   = OptionBuilder.withArgName( "driftMagnitude3" ).hasArg().withDescription("Drfit Magnitude2" ).create( "driftMagnitude3" );
		
		Option driftDelta   = OptionBuilder.withArgName( "driftDelta" ).hasArg().withDescription("driftDelta" ).create( "driftDelta" );

		Option driftNAttributes   = OptionBuilder.withArgName( "driftNAttributes" ).hasArg().withDescription("Drfit Number of Attributes" ).create( "driftNAttributes" );
		Option driftNAttributesValues   = OptionBuilder.withArgName( "driftNAttributesValues" ).hasArg().withDescription("Drfit Number of Attributes Values" ).create( "driftNAttributesValues" );

		Option totalNInstancesDuringDrift   = OptionBuilder.withArgName( "totalNInstancesDuringDrift" ).hasArg().withDescription("Drift duration in terms of no. of data points" ).create( "totalNInstancesDuringDrift" );

		Option discretizeOutOfCore   = new Option( "discretizeOutOfCore", "Discretize Out of core" );
		Option preProcessParameter   = OptionBuilder.withArgName( "preProcessParameter" ).hasArg().withDescription("preProcessParameter: <Explore, CreateHeader, Discretize, Slice, Normalize, MissingImputate>" ).create( "preProcessParameter" );

		Option ignoreAttributes   = OptionBuilder.withArgName( "ignoreAttributes" ).hasArg().withDescription("ignoreAttributes. {3,5,10}" ).create( "ignoreAttributes" );
		Option classAttribute   = OptionBuilder.withArgName( "classAttribute" ).hasArg().withDescription("classAttribute. E.g 2 (default is the last parameter" ).create( "classAttribute" );

		Option attributeType   = OptionBuilder.withArgName( "attributeType" ).hasArg().withDescription("attributeType. E.g {1,0,0,0,0,1,1}" ).create( "attributeType" );

		Option isGenerateDriftData = new Option( "isGenerateDriftData", "Generate Drift Data or work on provided file" );
		Option numClasses   = OptionBuilder.withArgName( "numClasses" ).hasArg().withDescription("Number of Classes" ).create( "numClasses" );
		Option numRandAttributes   = OptionBuilder.withArgName( "numRandAttributes" ).hasArg().withDescription("Number of Random Attributes" ).create( "numRandAttributes" );
		
		Option dicedStratified = new Option( "dicedStratified", "Do a stratified sampling (Dicing)" );
		Option dicedPercentage   = OptionBuilder.withArgName( "dicedPercentage" ).hasArg().withDescription("Drfit Percentage (0.2, 0.5, etc.)" ).create( "dicedPercentage" );
		Option dicedAt   = OptionBuilder.withArgName( "dicedAt" ).hasArg().withDescription("1000, 2000, etc" ).create( "dicedAt" );

		Option latentK   = OptionBuilder.withArgName( "latentK" ).hasArg().withDescription("latentK. E.g {10, 100, 500}" ).create( "latentK" );
		
		Option plotRMSEResuts = new Option( "plotRMSEResuts", "Plot RMSE Results" );
		
		options = new Options();

		options.addOption( help );
		options.addOption( version );
		options.addOption( verbose );

		options.addOption( experimentType );
		options.addOption( trainFile );
		options.addOption( testFile );
		options.addOption( cvFile );

		options.addOption( arffBufferSize );
		options.addOption( bufferSize );

		options.addOption( numExp );
		options.addOption( numFolds );

		options.addOption( discretization );
		options.addOption( discretizationParameter );
		options.addOption( normalizeNumeric );

		options.addOption( holdoutsetPrecentage );

		options.addOption( model );
		options.addOption( level );
		options.addOption( doWanbiac );
		options.addOption( doSelectiveKDB );
		options.addOption( doRegularization );
		options.addOption( lambda );
		options.addOption( adaptiveRegularization );

		options.addOption( optimizer );
		options.addOption( numIterations );
		options.addOption( sgdType );
		options.addOption( sgdTuningParameter );
		options.addOption( doCrossValidateTuningParameter );

		options.addOption( mcmcType );

		options.addOption( objectiveFunction );
		options.addOption( doBinaryClassification );
		options.addOption( dataStructureParameter );
		options.addOption( hashParameter );

		options.addOption( doDiscriminative );

		options.addOption( adaptiveControl );
		options.addOption( adaptiveControlParameter );

		options.addOption( prequentialOutputResolution );

		options.addOption( prequentialBufferOutputResolution );
		options.addOption( doMovingAverage );

		options.addOption( tempDirectory );
		options.addOption( ouputResultsDirectory );

		options.addOption( flowParameter );
		options.addOption( isGenerateDriftData );
		options.addOption( driftType );
		options.addOption( driftMagnitude );
		options.addOption( driftMagnitude2 );
		options.addOption( driftMagnitude3 );

		options.addOption( driftNAttributes );
		options.addOption( driftNAttributesValues );

		options.addOption( featureSelection );
		options.addOption( featureSelectionParameter );

		options.addOption( totalNInstancesDuringDrift );

		options.addOption( discretizeOutOfCore );
		options.addOption( preProcessParameter );

		options.addOption( ignoreAttributes );
		options.addOption( classAttribute );

		options.addOption( attributeType );

		options.addOption( dicedStratified );
		options.addOption( dicedPercentage );
		options.addOption( dicedAt );
		
		options.addOption( numClasses );
		
		options.addOption( numRandAttributes );
		
		options.addOption(  latentK );
		options.addOption(  driftDelta );
		
		options.addOption(  plotRMSEResuts );
	}

	public void setOptions(String[] args) {

		CommandLine line = null;
		CommandLineParser parser = new DefaultParser();
		try {
			line = parser.parse( options, args );
		}
		catch( ParseException exp ) {
			System.err.println( "Parsing failed.  Reason: " + exp.getMessage() );
		}

		if (line.hasOption("help")) {
			Globals.printHelp();
			System.exit(0);
		}

		if (line.hasOption("verbose")) { 
			//Globals.setVerbose(Boolean.parseBoolean(line.getOptionValue( "verbose" )));
			Globals.setVerbose(true);
		}

		if (line.hasOption("normalizeNumeric")) { 
			//Globals.setNormalizeNumeric(Boolean.parseBoolean(line.getOptionValue( "normalizeNumeric" )));
			Globals.setNormalizeNumeric(true);
		}

		if ( line.hasOption( "discretization" ) )  Globals.setDiscretization(line.getOptionValue( "discretization" ));
		if ( line.hasOption( "discretizationParameter" ) )  Globals.setDiscretizationParameter(Integer.parseInt(line.getOptionValue( "discretizationParameter" )));

		if ( line.hasOption( "holdoutsetPrecentage" ) )  Globals.setHoldoutsetPrecentage(Integer.parseInt(line.getOptionValue( "holdoutsetPrecentage" )));

		if ( line.hasOption( "experimentType" ) )  
			Globals.setExperimentType(line.getOptionValue( "experimentType" ));

		if ( line.hasOption( "trainFile" ) )  Globals.setTrainFile(line.getOptionValue( "trainFile" ));
		if ( line.hasOption( "testFile" ) )  Globals.setTestFile(line.getOptionValue( "testFile" ));
		if ( line.hasOption( "cvFile" ) )  Globals.setCvFile(line.getOptionValue( "cvFile" ));
		if ( line.hasOption( "numExp" ) )   
			Globals.setNumExp(Integer.parseInt(line.getOptionValue( "numExp" )));

		if ( line.hasOption( "arffBufferSize" ) )  Globals.setARFF_BUFFER_SIZE(Integer.parseInt(line.getOptionValue( "arffBufferSize" )));
		if ( line.hasOption( "bufferSize" ) )  Globals.setBUFFER_SIZE(Integer.parseInt(line.getOptionValue( "bufferSize" )));

		if ( line.hasOption( "numFolds" ) )  
			Globals.setNumFolds(Integer.parseInt(line.getOptionValue( "numFolds" )));

		if ( line.hasOption( "model" ) )  Globals.setModel(line.getOptionValue( "model" ));
		if ( line.hasOption( "level" ) )  Globals.setLevel(Integer.parseInt(line.getOptionValue( "level" )));

		if ( line.hasOption( "doWanbiac" ) )  {
			//Globals.setDoWanbiac(Boolean.parseBoolean(line.getOptionValue( "doWanbiac" )));
			Globals.setDoWanbiac(true);
		}
		
		if ( line.hasOption( "doSelectiveKDB" ) ) {
			//Globals.setDoSelectiveKDB(Boolean.parseBoolean(line.getOptionValue( "doSelectiveKDB" )));
			Globals.setDoSelectiveKDB(true);
		}

		if ( line.hasOption( "doRegularization" ) )  {
			//Globals.setDoRegularization(Boolean.parseBoolean(line.getOptionValue( "doRegularization" )));
			Globals.setDoRegularization(true);
		}

		if ( line.hasOption( "lambda" ) )  
			Globals.setLambda(Double.parseDouble(line.getOptionValue( "lambda" )));

		if ( line.hasOption( "adaptiveRegularization" ) )  Globals.setAdaptiveRegularization(line.getOptionValue( "adaptiveRegularization" ));
		if ( line.hasOption( "regularizationType" ) )  Globals.setRegularizationType(line.getOptionValue( "regularizationType" ));
		if ( line.hasOption( "regularizationTowards" ) )  Globals.setRegularizationTowards(Double.parseDouble(line.getOptionValue( "regularizationTowards" )));

		if ( line.hasOption( "optimizer" ) )  Globals.setOptimizer(line.getOptionValue( "optimizer" ));
		if ( line.hasOption( "numIterations" ) )  Globals.setNumIterations(Integer.parseInt(line.getOptionValue( "numIterations" )));
		if ( line.hasOption( "sgdType" ) )  Globals.setSgdType(line.getOptionValue( "sgdType" ));
		if ( line.hasOption( "sgdTuningParameter" ) )  Globals.setSgdTuningParameter(Double.parseDouble(line.getOptionValue( "sgdTuningParameter" )));

		if ( line.hasOption( "doCrossValidateTuningParameter" ) )  {
			//Globals.setDoCrossValidateTuningParameter(Boolean.parseBoolean(line.getOptionValue( "doCrossValidateTuningParameter" )));
			Globals.setDoCrossValidateTuningParameter(true);
		}

		if ( line.hasOption( "mcmcType" ) )  Globals.setMcmcType(line.getOptionValue( "mcmcType" ));

		if ( line.hasOption( "featureSelection" ) )  Globals.setFeatureSelection(line.getOptionValue( "featureSelection" ));
		if ( line.hasOption( "featureSelectionParameter" ) )  Globals.setFeatureSelectionParameter(Double.parseDouble(line.getOptionValue( "featureSelectionParameter" )));


		if ( line.hasOption( "doBinaryClassification" ) )  {
			Globals.setDoBinaryClassification(true);
		}

		if ( line.hasOption( "objectiveFunction" ) )  Globals.setObjectiveFunction(line.getOptionValue( "objectiveFunction" ));
		if ( line.hasOption( "dataStructureParameter" ) )  
			Globals.setDataStructureParameter(line.getOptionValue( "dataStructureParameter" ));

		if ( line.hasOption( "hashParameter" ) )  Globals.setHashParameter(Integer.parseInt(line.getOptionValue( "hashParameter" )));

		if ( line.hasOption( "doDiscriminative" ) )  { 
			Globals.setDoDiscriminative(true);
		}

		if ( line.hasOption( "adaptiveControl" ) )  Globals.setAdaptiveControl(line.getOptionValue( "adaptiveControl" ));
		if ( line.hasOption( "adaptiveControlParameter" ) )  Globals.setAdaptiveControlParameter(Double.parseDouble(line.getOptionValue( "adaptiveControlParameter" )));

		if ( line.hasOption( "prequentialOutputResolution" ) )  Globals.setPrequentialOutputResolution(Integer.parseInt(line.getOptionValue( "prequentialOutputResolution" )));

		if ( line.hasOption( "doMovingAverage" ) )  { 
			Globals.setDoMovingAverage(true);
		}

		if ( line.hasOption( "prequentialBufferOutputResolution" ) )  
			Globals.setPrequentialBufferOutputResolution(Integer.parseInt(line.getOptionValue( "prequentialBufferOutputResolution" )));

		if ( line.hasOption( "tempDirectory" ) )  Globals.setTempDirectory(line.getOptionValue( "tempDirectory" ));
		if ( line.hasOption( "ouputResultsDirectory" ) )  
			Globals.setOuputResultsDirectory(line.getOptionValue( "ouputResultsDirectory" ));


		if ( line.hasOption( "flowParameter" ) )  Globals.setFlowParameter(line.getOptionValue( "flowParameter" ));

		if ( line.hasOption( "driftType" ) )  
			Globals.setDriftType(line.getOptionValue( "driftType" ));
		
		if ( line.hasOption( "isGenerateDriftData" ) )  {
			Globals.setGenerateDriftData(true);
		}

		if ( line.hasOption( "driftMagnitude" ) )  
			Globals.setDriftMagnitude(Double.parseDouble(line.getOptionValue( "driftMagnitude" )));
		if ( line.hasOption( "driftMagnitude2" ) ) 
			Globals.setDriftMagnitude2(Integer.parseInt(line.getOptionValue( "driftMagnitude2" )));
		if ( line.hasOption( "driftMagnitude3" ) ) 
			Globals.setDriftMagnitude3(Double.parseDouble(line.getOptionValue( "driftMagnitude3" )));
		
		if ( line.hasOption( "driftDelta" ) )  
			Globals.setDriftDelta(Double.parseDouble(line.getOptionValue( "driftDelta" )));

		if ( line.hasOption( "driftNAttributes" ) )  Globals.setDriftNAttributes(Integer.parseInt(line.getOptionValue( "driftNAttributes" )));
		if ( line.hasOption( "driftNAttributesValues" ) )  Globals.setDriftNAttributesValues(Integer.parseInt(line.getOptionValue( "driftNAttributesValues" )));
		if ( line.hasOption( "totalNInstancesDuringDrift" ) ) 
			Globals.setTotalNInstancesDuringDrift(Integer.parseInt(line.getOptionValue( "totalNInstancesDuringDrift" )));

		if ( line.hasOption( "discretizeOutOfCore" ) )  { 
			Globals.setDiscretizeOutOfCore(true);
		}

		if ( line.hasOption( "preProcessParameter" ) )  Globals.setPreProcessParameter(line.getOptionValue( "preProcessParameter" ));
		if ( line.hasOption( "ignoreAttributes" ) )  Globals.setIgnoreAttributes(line.getOptionValue( "ignoreAttributes" ));

		if ( line.hasOption( "classAttribute" ) )  Globals.setClassAttribute(Integer.parseInt(line.getOptionValue( "classAttribute" )));

		if ( line.hasOption( "attributeType" ) )  Globals.setAttributeType((line.getOptionValue( "attributeType" )));

		if ( line.hasOption( "dicedStratified" ) )  { 
			Globals.setDicedStratified(true);
		}

		if ( line.hasOption( "dicedAt" ) )  Globals.setDicedAt(Integer.parseInt(line.getOptionValue( "dicedAt" )));

		if ( line.hasOption( "dicedPercentage" ) )  Globals.setDicedPercentage(Integer.parseInt(line.getOptionValue( "dicedPercentage" )));
		
		if ( line.hasOption( "numClasses" ) )  Globals.setNumClasses(Integer.parseInt(line.getOptionValue( "numClasses" )));

		if ( line.hasOption( "numRandAttributes" ) )  Globals.setNumRandAttributes(Integer.parseInt(line.getOptionValue( "numRandAttributes" )));
		
		if ( line.hasOption( "latentK" ) )  Globals.setLatentK(Integer.parseInt(line.getOptionValue( "latentK" )));
		
		if ( line.hasOption( "plotRMSEResuts" ) )  { 
			Globals.setPlotRMSEResuts(true);
		}
		 
	}

	public void getOptions() {

	}

}
