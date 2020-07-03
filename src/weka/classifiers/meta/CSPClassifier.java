package weka.classifiers.meta;

import weka.classifiers.Classifier;
import weka.classifiers.RandomizableMultipleClassifiersCombiner;
import weka.classifiers.UpdateableClassifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.M5P;
import weka.clusterers.Clusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.util.*;


public class CSPClassifier extends RandomizableMultipleClassifiersCombiner implements UpdateableClassifier {

    /*** Subproblem maker used to divide the dataset into instance groups for dedicated training. */
    protected Clusterer subProblemMaker;

    /*** The subproblem selector is a classifier used to select the most adequate model for predict instances. */
    protected Classifier subProblemSelector;

    /*** Subproblems (a vector of instance groups). */
    protected Instances subproblems[];

    /*** Dataset which contains an attribute with the subproblem assignments. */
    protected Instances subproblem_assignments;

    /*** List of base classifers trained for each subproblem  */
    protected List<Classifier> subclassifiers;

    /** Inherited attribute: m_Classifiers (* the base classifiers class specification *) */

    /*** The number of subproblems. */
    protected int numSubproblems;



    public CSPClassifier() {
        super();
        subProblemMaker = new SimpleKMeans();
        subProblemSelector = new J48();
        m_Classifiers = new Classifier[1];
        m_Classifiers[0] = new M5P();
        subclassifiers = new ArrayList<>();
        numSubproblems = 2;
        //subproblems = new Instances[numSubproblems];
    }

    public CSPClassifier(Clusterer subProblemMaker, Classifier subProblemSelector) {
        this.subProblemMaker = subProblemMaker;
        this.subProblemSelector = subProblemSelector;
    }

    @Override
    public Enumeration<Option> listOptions() {
        Vector<Option> newVector = new Vector<Option>(2);

        newVector.addElement(new Option(
                "\tSub Problem Maker.\n"
                        + "\t(default SimpleKMeans)",
                "M", 1, "-M <clusterer>"));
        newVector.addElement(new Option(
                "\tSub Problem Selector.\n"
                        + "\t(default J48)",
                "S", 1, "-S <classifier>"));
        newVector.addElement(new Option(
                "\tNumber of subproblems.\n"
                        + "\t(default 2)",
                "N", 1, "-N <int>"));

        newVector.addAll(Collections.list(super.listOptions()));

        return newVector.elements();
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        String temp = Utils.getOption('M', options);
        this.subProblemMaker = (Clusterer) Class.forName(temp).newInstance();

        temp = Utils.getOption('S', options);
        this.subProblemSelector = (Classifier) Class.forName(temp).newInstance();

        temp = Utils.getOption('N', options);
        this.numSubproblems = Integer.parseInt(temp);

        super.setOptions(options);
    }

    @Override
    public String[] getOptions() {
        Vector<String> options = new Vector<String>();

        options.add("-M");
        options.add("" + subProblemMaker.getClass().getName());

        options.add("-S");
        options.add("" + subProblemSelector.getClass().getName());

        options.add("-N");
        options.add("" + this.numSubproblems);

        Collections.addAll(options, super.getOptions());

        return options.toArray(new String[0]);
    }

    @Override
    protected String getClassifierSpec(int index) {
        return super.getClassifierSpec(index);
    }

    @Override
    public String seedTipText() {
        return super.seedTipText();
    }

    @Override
    public void setSeed(int seed) {
        super.setSeed(seed);
    }

    @Override
    public int getSeed() {
        return super.getSeed();
    }

    public int getNumSubproblems() {
        return numSubproblems;
    }

    public void setNumSubproblems(int numSubproblems) {
        this.numSubproblems = numSubproblems;
    }

    public String numSubproblemsTipText() {
        return "The number of subproblems.";
    }

    @Override
    public String classifiersTipText() {
        return "The classifiers used to train the different sub problems (groups of instances).";
    }

    @Override
    public void setClassifiers(Classifier[] classifiers) {
        super.setClassifiers(classifiers);
    }

    @Override
    public Classifier[] getClassifiers() {
        return super.getClassifiers();
    }

    @Override
    public Classifier getClassifier(int index) {
        return super.getClassifier(index);
    }

    @Override
    public void preExecution() throws Exception {
        super.preExecution();
        System.out.println("PRE-EXECUTION!");
    }

    @Override
    public void postExecution() throws Exception {
        super.postExecution();
        System.out.println("POST-EXECUTION!");
    }

    @Override
    public String getRevision() {
        return "REVISION GUALBE: " + super.getRevision();
    }


    @Override
    public void buildClassifier(Instances data) throws Exception {
        int i = 0;
        final int c_index = data.classIndex();

        // Test capabilities of the base classifier/s.
        try{
            m_Classifiers[0].getCapabilities().testWithFail(data);
        } catch (Exception e){
            throw new Exception("The base classifier (" + m_Classifiers[0].getClass() + ") cannot handle this dataset. Please choose another.");
        }

        // Set preserve instance order.
//        Filter filter = this.m_Filter;
//        filter.setInputFormat(instances);
//        Filter.useFilter(instances, filter);
//        this.puntos_base = ((XNNFilter)filter).getPuntosBase();
//        this.claseContAsoc = ((XNNFilter)filter).getClaseContAsoc();

        // Saving data to set the clusters buckets.
        subproblem_assignments = new Instances(data);

        // Remove class from the dataset "subproblem_assignments".
        final Remove removeFilter = new Remove();
        removeFilter.setAttributeIndicesArray(new int[]{subproblem_assignments.classIndex()});
        removeFilter.setInputFormat(subproblem_assignments);
        subproblem_assignments = Filter.useFilter(subproblem_assignments, removeFilter);

        // Train the clusterer.
        subProblemMaker.buildClusterer(subproblem_assignments);
        subproblems = new Instances[subProblemMaker.numberOfClusters()];

        // Append the cluster number as new class.
        int n = subProblemMaker.numberOfClusters();
        List<String> classValues = new ArrayList<String>(n);
        for (i = 0; i < n; i++)
            classValues.add(String.valueOf(i));
        //List<Integer> range = IntStream.rangeClosed(0, subProblemMaker.numberOfClusters()-1).boxed().collect(Collectors.toList());
        subproblem_assignments.insertAttributeAt(new Attribute("subproblem", classValues), subproblem_assignments.numAttributes());
        subproblem_assignments.setClassIndex(subproblem_assignments.numAttributes()-1);

        // Initialize subproblems
        for(i = 0; i < subproblems.length; i++) {
            subproblems[i] = new Instances(data, -1);
        }

        // Obtain the cluster array.
        final int[] clust = ((SimpleKMeans) subProblemMaker).getAssignments();

        // Set the class index in all instances of the dataset.
        i = 0;
        for(final Instance instance : subproblem_assignments) {
            instance.setClassValue(Integer.toString(clust[i++]));
        }

        // Store instances for each subproblem.
        i = 0;
        for(final Instance instance : data) {
            subproblems[clust[i++]].add(instance);
        }

        try{
            subProblemSelector.getCapabilities().testWithFail(subproblem_assignments);
        } catch (Exception e){
            throw new Exception("The SubProblem Selector classifier (" + subProblemSelector.getClass() + ") cannot handle this dataset. Please choose another.");
        }

        // Train the SubProblem Selector.
        subProblemSelector.buildClassifier(subproblem_assignments);

        // Train the subproblem classifiers
        for (final Instances bucket : subproblems) {
            final Classifier classifier = m_Classifiers[0].getClass().newInstance();
            classifier.buildClassifier(bucket);
            subclassifiers.add(classifier);
        }

    }

    @Override
    public String toString() {

        if (this.subproblems == null)
            return "Not initialized";

        int n = this.subproblems.length;
        String res = "";

        // Subproblems Basic Statistics.
        res += "\nSubproblems: " + n + "\n";
        for (int i = 0; i < n; i++) {
            res += "Subproblem #" + (i+1) + ": " + this.subproblems[i].numInstances() + " instances.\n";
        }

        // Model evaluation.
        Evaluation eval = null;
        Random rand = new Random(this.m_Seed);
        int folds = 10;
        try {
            // SubProblem Selector Evaluation.
            eval = new Evaluation(this.subproblem_assignments);
            eval.crossValidateModel(this.subProblemSelector, this.subproblem_assignments, folds, rand);
            res += "\nSubProblem Selector Model Evaluation:" + eval.toSummaryString() + "\n";

            // Base Classifiers Evaluation.
            for (int i = 0; i < n; i++) {
                eval = new Evaluation(this.subproblems[i]);
                eval.crossValidateModel(this.subclassifiers.get(i), this.subproblems[i], folds, rand);
                res += "\nEvaluation of the Model for SubProblem #" + (i+1) + ":" + eval.toSummaryString() + "\n";
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        return res;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        Instance copy = (Instance) instance.copy();
        copy.setDataset(this.subproblem_assignments);
        double estimated_subproblem = subProblemSelector.classifyInstance(copy);
        return subclassifiers.get((int)estimated_subproblem).distributionForInstance(instance);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        Instance copy = (Instance) instance.copy();
        copy.setDataset(this.subproblem_assignments);
        double estimated_subproblem = subProblemSelector.classifyInstance(copy);
        return subclassifiers.get((int)estimated_subproblem).classifyInstance(instance);
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities res = new Capabilities(this);
        res.enableAll();

        return res;
    }

    public Clusterer getSubProblemMaker() {
        return this.subProblemMaker;
    }

    public void setSubProblemMaker(Clusterer clusterer) {
        this.subProblemMaker = clusterer;
    }

    public String subProblemMakerTipText() {
        return "Subproblem maker used to divide the dataset into instance groups for dedicated training.";
    }

    public Classifier getSubProblemSelector() {
        return subProblemSelector;
    }

    public void setSubProblemSelector(Classifier subProblemSelector) {
        this.subProblemSelector = subProblemSelector;
    }

    public String subProblemSelectorTipText() {
        return "The subproblem selector is a classifier used to select the most adequate model for predict instances.";
    }

    @Override
    public void updateClassifier(Instance instance) throws Exception {
        Instance copy = (Instance) instance.copy();
        copy.setDataset(this.subproblem_assignments);
        double estimated_subproblem = subProblemSelector.classifyInstance(copy);
        ((UpdateableClassifier)subclassifiers.get((int)estimated_subproblem)).updateClassifier(instance);
        System.out.println("\nASPCLASSIFIER --> UPDATECLASSIFIER\n");
    }
}
