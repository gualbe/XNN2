package weka.filters.supervised.attribute;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.SimpleBatchFilter;
import weka.filters.supervised.attribute.xnn.EnhancementModel;
import weka.filters.supervised.attribute.xnn.ProbabilityEnhancementModel;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class XNNShredding  extends SimpleBatchFilter {

    /**
     * Modelo de mejora de clase.
     */
    protected EnhancementModel enhancementModel = new ProbabilityEnhancementModel();

    /**
     * Tuplas de números reales asociadas a las instancias cuando existe
     * etiqueta nominal.-
     * <p>
     * Se trata de una n-tupla de números reales por cada instancia
     * que conforma la clase continua asociada a la etiqueta nominal.
     */
    protected Map<String, double[][]> enhancementVectors;


    @Override
    public String globalInfo() {
        return "A filter to balace classes in unbalances contexts using ProbabilityEnhacementModel. "
                + "This filter divide de majority class into several subclasses with similar size as minorityClass."
                + "Also removes not relevant information based on local information."
                + "\n\nGualberto Asencio-Cortes, Ph.D. (guaasecor@upo.es)\n";
    }

    @Override
    protected Instances determineOutputFormat(Instances instances) {

        return new Instances(instances, 0);
    }

    @Override
    protected Instances process(Instances instances) throws Exception {

        /*
         * Variables initialization
         */
        assert instances.classAttribute().numValues()==2 : "El número de clases debe ser 2";
        Instances result, enhancedClasses, enhancedClassesDiscretized;
        Map<String, Long> classDistr = new HashMap<>();
        String[] keys, attributeValues;
        String majorityClass, minorityClass, discrClass;
        long nIrrelevantInstances, majorityClassSelected;
        int nSamplesIrrelevantInstances, majorityClassIndx, minorityClassIdx, numBins, z;
        double prob, majorityClassSubtype;
        List<Integer> selectedInstancesIndexes;
        double[][] distToBase, classEnhanced;
        List<double[]> enhancedSelectedClasses;
        ArrayList<Attribute> enhancedClassesAttributes;
        Map<String, Double> discrToValue;
        double[] newClasses, row;
        Attribute classAttr;

        // Initialize result
        result = new Instances(determineOutputFormat(instances), 0);

        /*
         * Use probability enhacement method to obtain enhaced classes and distance to base
         */

        enhancementModel.setInstances(instances);
        enhancementVectors = enhancementModel.generateEnhancementVectors();
        distToBase = enhancementVectors.get("XNNDistToBase");
        classEnhanced = enhancementVectors.get("XNNProbs");

        /*
         * First step: Irrelevant instances reduction
         */

        // First calculate the distribution of samples by class in data
        for(int i=0; i<instances.classAttribute().numValues(); i++){
            String cls = instances.classAttribute().value(i);
            Long nInstancesCls = instances.stream().filter(ins->ins.stringValue(instances.classAttribute()).equals(cls)).count();
            classDistr.put(cls, nInstancesCls);
        }
        keys = classDistr.keySet().toArray(new String[0]);

        // Set the majority and minority class strings value for future
        majorityClass =  classDistr.get(keys[0])>classDistr.get(keys[1]) ?  keys[0]: keys[1];
        minorityClass =  classDistr.get(keys[0])<classDistr.get(keys[1]) ?  keys[0]: keys[1];

        // Count the instances from the majority class that not give relevant information
        nIrrelevantInstances = IntStream.range(0, instances.numInstances())
                .filter(i->distToBase[i][0]==0 && instances.get(i).stringValue(instances.classAttribute()).equals(majorityClass))
                .count();

        // Calculate the minimal representative samples by Yamane formula
        nSamplesIrrelevantInstances = (int) Math.round(nIrrelevantInstances/(1+nIrrelevantInstances*(Math.pow(0.05,2))));

        // Set the probability of select a random irrelevant sample
        prob = (double) nSamplesIrrelevantInstances/nIrrelevantInstances;

        // Select the instances from minority class, majority class with distance to base > 0 and a random sample from majority class
        selectedInstancesIndexes = IntStream.range(0, instances.numInstances())
                .filter(i->instances.get(i).stringValue(instances.classAttribute()).equals(minorityClass) ||
                    (instances.get(i).stringValue(instances.classAttribute()).equals(majorityClass) && Math.random()<=prob)
                    || distToBase[i][0]>0)
            .boxed().collect(Collectors.toList());

        // Obtain the enhanced classes from selected instances
        enhancedSelectedClasses = selectedInstancesIndexes.stream()
                        .map(i->classEnhanced[i]).collect(Collectors.toList());

        /*
         * Second step: Discretize enhanced classes and set to the original data
         */

        // Initialize the attributes from the enhanced classes instances
        enhancedClassesAttributes = new ArrayList<>();
        enhancedClassesAttributes.add(new Attribute("majorityClass"));
        enhancedClassesAttributes.add(new Attribute("minorityClass"));

        // Initialize some indexes to set the majority class always first
        if(majorityClass.equals(keys[0])){
            majorityClassIndx = 0;
            minorityClassIdx = 0;
        }else{
            majorityClassIndx = 2;
            minorityClassIdx = 1;
        }

        // Initialize the enhanced classes instances
        enhancedClasses = new Instances("enhancedClasses", enhancedClassesAttributes, enhancedSelectedClasses.size());

        // Set the data values of the classes
        for(double[] value : enhancedSelectedClasses){
            enhancedClasses.add(new DenseInstance(1, IntStream.range(0, 2).map(i-> Math.abs(majorityClassIndx-i-minorityClassIdx))
                    .mapToDouble(v->value[v]*1e6).toArray()));// Note that the majority class is always at column 0
        }

        // Number of majority class instances after sampling
        majorityClassSelected = selectedInstancesIndexes.size() - classDistr.get(minorityClass);

        // Calculate the number of bind for discretization dividing the number of majority class instances by the number minority class intances
        numBins = (int) Math.round((double) majorityClassSelected/classDistr.get(minorityClass));

        // Initialize the discretization with equal frequency filter for the enhances classes instance
        weka.filters.unsupervised.attribute.Discretize dis = new weka.filters.unsupervised.attribute.Discretize();
        dis.setDesiredWeightOfInstancesPerInterval(-1.0D);
        dis.setBins(numBins);
        dis.setAttributeIndices("first-last");
        dis.setUseEqualFrequency(true);
        dis.setInputFormat(enhancedClasses);

        // Execute the discretization filter
        enhancedClassesDiscretized = Filter.useFilter(enhancedClasses, dis);

        majorityClassSubtype = 1.;
        discrToValue = new HashMap<>();
        newClasses = new double[enhancedClasses.numInstances()];

        // Calculate the double representation of the discretized classes
        for(int i=0;i<enhancedClassesDiscretized.numInstances(); i++){
            discrClass = enhancedClassesDiscretized.get(i).stringValue(0);

            if(discrToValue.containsKey(discrClass)){
                newClasses[i] = discrToValue.get(discrClass);
            }else if(enhancedClasses.get(i).value(0)>enhancedClasses.get(i).value(1)){ //Only add/set new value if is majority class
                newClasses[i] = majorityClassSubtype;
                discrToValue.put(discrClass, majorityClassSubtype++);
            }
        }

        // Calculate the new class attribute values
        attributeValues = new String[discrToValue.keySet().size()+1];
        attributeValues[0] = minorityClass;
        for(int j=1; j<attributeValues.length; j++){
            attributeValues[j] = majorityClass+"_"+j;
        }
        // Replace the class by the new enhances class attribute
        classAttr = new Attribute(instances.classAttribute().name(), Arrays.asList(attributeValues));
        result.replaceAttributeAt(classAttr, instances.classIndex());

        z = 0;
        // Create the new data and replace the majority class values by its double representation and set minority class to 0
        for(Integer selInsIdx : selectedInstancesIndexes){

            row =  new double[instances.numAttributes()];

            for(int att=0; att<instances.numAttributes(); att++){
                if(att==instances.classIndex()){
                    if(instances.get(selInsIdx).stringValue(att).equals(minorityClass)){
                        row[att] = 0.;
                    }else{
                        row[att] = newClasses[z];
                    }

                }else{
                    row[att] = instances.get(selInsIdx).value(att);
                }
            }
            z++;

            result.add(new DenseInstance(1, row));
        }

        // Update the classIndex
        result.setClassIndex(instances.classIndex());
        //Update the output format to the new class attribute format
        setOutputFormat(result);

        return result;
    }
}
