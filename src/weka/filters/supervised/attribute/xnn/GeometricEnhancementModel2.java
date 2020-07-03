package weka.filters.supervised.attribute.xnn;


import weka.core.*;
import weka.filters.supervised.attribute.xnn.utils.XNNUtils;

import java.util.*;


public class GeometricEnhancementModel2 extends GeometricEnhancementModel implements OptionHandler {

    private static final double EPSILON =  0.00001;
    private int qVersion;
    private int k1 ;
    private int k2;
    private double wtt;
    private double wtnt;
    private int summaryFunction;
    private double minkowskiDegree;




    public GeometricEnhancementModel2() {
        this(3, 3, 1, 1.0, 1.0, 1, 1, false, false, false, false);
    }

    public GeometricEnhancementModel2(int k1, int k2, int qVersion, double wtt, double wtnt, int summaryFunction, double minkowsky_d, boolean noNorm, boolean m_Debug, boolean toFileDebug, boolean preprocessDebug) {
        super(noNorm, m_Debug, toFileDebug);
        this.k1 = k1;
        this.k2 = k2;
        this.qVersion = qVersion;
        this.wtt = wtt;
        this.wtnt = wtnt;
        this.summaryFunction = summaryFunction;
        this.minkowskiDegree = minkowsky_d;
    }

    public SelectedTag getQVersion() {
        return new SelectedTag(qVersion, GeometricEnhancementModel.Q_VERSION);
    }

    public void setQVersion(SelectedTag newQVersion) {
        if (newQVersion.getTags() == GeometricEnhancementModel.Q_VERSION) {
            this.qVersion = newQVersion.getSelectedTag().getID();
        }
    }

    public int getK1() {
        return k1;
    }

    public void setK1(int k1) {
        this.k1 = k1;
    }

    public int getK2() {
        return k2;
    }

    public void setK2(int k2) {
        this.k2 = k2;
    }

    public double getWtt() {
        return wtt;
    }

    public void setWtt(double wtt) {
        this.wtt = wtt;
    }

    public double getWtnt() {
        return wtnt;
    }

    public void setWtnt(double wtnt) {
        this.wtnt = wtnt;
    }

    public SelectedTag getSummaryFunction() {
        return new SelectedTag(summaryFunction, GeometricEnhancementModel.SUMMARY_FUNC);
    }

    public void setSummaryFunction(SelectedTag newSummFunc) {
        if (newSummFunc.getTags() == GeometricEnhancementModel.SUMMARY_FUNC) {
            this.summaryFunction = newSummFunc.getSelectedTag().getID();
        }
    }

    public double getMinkowskiDegree() {
        return minkowskiDegree;
    }

    public void setMinkowskiDegree(double minkowskiDegree) {
        this.minkowskiDegree = minkowskiDegree;
    }

    @Override
    public Map<String, double[][]> generateEnhancementVectors() {

        double [][] xnnpoints = new double[instances.numInstances()][numValoresClase - 1]; // + Num. extra information attributes.
        double [][] distToBase = new double[instances.numInstances()][1];
        double [][] sdToForeign = new double[instances.numInstances()][1];
        Map<String, double[][]> res = Collections.synchronizedMap(new LinkedHashMap<String, double[][]>());

        double [][] p;
        double [] q, d, f;
        double accum;
        int i, j, k, n, clase_g;
        Instance g;

        if (m_Debug) {
            escribirLog("Construcción del modelo continuo...\n");
            escribirLog("Las etiquetas están numeradas de 0 a " + (numValoresClase - 1) + "\n");
        }

        if (m_Debug)
            escribirLog("Construcción de los puntos base...\n");

        GeometricEnhancementModel.getBasePoints(numValoresClase);

        if (m_Debug)
            presentar_puntos_base();

        if (m_Debug)
            escribirLog("Extendiendo semántica en las instancias...\n");

        i = 0;
        n = instances.numInstances();
        while (i < n){
            g = instances.instance(i);
            clase_g = (int) g.classValue();
            if (m_Debug)
                escribirLog("Procesando instancia " + i + ", etiqueta = " + clase_g + "  {\n");

            // Obtener Q
            q = construir_punto_q(g);

            // Obtener puntos P
            if (m_Debug)
                escribirLog("\tCalculando puntos P  {\n");
            p = new double[numValoresClase - 1][numValoresClase - 1];
            d = new double[numValoresClase -1];
            k = 0;
            for (j = 0; j < numValoresClase; j++){
                if( j != clase_g) {
                    p[k] = calcularP(q, g, j);
                    d[k] = getDistanceNearestNeighbors(g, j);
                    if(d[k]==1.)
                        d[k] -= EPSILON;
                    if (m_Debug)
                        escribirLog("\t\tT_" + j + " = " + d[k] + " --> P_" + j + " = " + XNNUtils.vector2string(p[k]) + "\n");
                    k ++;
                }
            }
            if (m_Debug)
                escribirLog("\t}\n");

            // Calcular punto F
            f = new double[numValoresClase - 1];
            for (j = 0; j < numValoresClase - 1; j++){
                accum = 0.;
                for ( k = 0; k < p.length; k++) {
                    f[j] += (1 - d[k]) * p[k][j];
                    accum += (1 - d[k]);
                }

                f[j] = f[j] / accum;
            }

            // Guardar XNNPoint.
            xnnpoints[i] = f;

            // Guardar distToBase.
            distToBase[i][0] = XNNUtils.distance(f, basePoints[clase_g]);

            // Guardar sdToForeign.
            sdToForeign[i][0] = stdev_dist_to_foreign_base_points(f, clase_g);

            i++;
            if (m_Debug)
                escribirLog("\tPunto F = " + XNNUtils.vector2string(xnnpoints[i-1]) + "\n");

            if (m_Debug)
                escribirLog("}\n");
        }

        // Construir tabla asociativa de resultado
        res.put("XNNpoints", xnnpoints);
        res.put("XNNDistToBase", distToBase);
        res.put("XNNSdForeign", sdToForeign);

        return res;
    }

    /**
     * EN DESUSO.
     */
    protected double [] appendExtraInformation (double [] v, int cl) {
        double [] res = new double [v.length + 2];
        int i;

        for (i = 0; i < v.length; i++)
            res[i] = v[i];

        res[i]  = XNNUtils.distance(v, basePoints[cl]);
        res[i+1] = stdev_dist_to_foreign_base_points(v, cl);

        return res;
    }

    private double stdev_dist_to_foreign_base_points(double [] v, int cl) {
        int i, n = basePoints.length;
        double mean = 0, res = 0;

        // Average distance to foreign base points.
        for (i = 0; i < n; i++)
            if (i != cl)
                mean += XNNUtils.distance(basePoints[i], v);
        mean /= n - 1;

        // Standard deviation.
        for (i = 0; i < n; i++)
            if (i != cl)
                res += Math.abs(mean - XNNUtils.distance(basePoints[i], v));
        res /= n - 1;

        return res;
    }

    @Override
    public Instances determineOutputFormat(Instances inputFormat) {
        Instances result = new Instances(inputFormat, 0);
        int n = inputFormat.numClasses();

        for (int i = 0; i < n - 1; i++) {
            result.insertAttributeAt(new Attribute("XNNPoint" + i), result.classIndex());
        }

        result.insertAttributeAt(new Attribute("XNNDistToBase"), result.classIndex());
        result.insertAttributeAt(new Attribute("XNNSdForeign"), result.classIndex());

        return result;
    }

    private double[] calcularP(double[] q, Instance base, int clase){
        double[] P, v, B1, B2;
        int i, clase_base;
        double lambda = 0., lambda_denominator = 0., TNT;

        clase_base = (int) base.classValue();
        P = new double[numValoresClase -1];
        v = new double[numValoresClase - 1];
        B1 = basePoints[clase_base];
        B2 = basePoints[clase];
        TNT = getDistanceNearestNeighbors(base, clase);

        for (i = 0; i < numValoresClase - 1; i++)
            v[i] = B2[i] - B1[i];

        for (i = 0; i < numValoresClase - 1; i++) {
            lambda +=  v[i] * (q[i] - B1[i]);
            lambda_denominator += v[i]*v[i];
        }

        lambda = lambda / lambda_denominator;

        for (i = 0; i < numValoresClase -1; i++){
            P[i] = B1[i] + lambda * v[i];
        }

        for (i = 0; i < numValoresClase -1; i++){
            P[i] = TNT * q[i] + (1 - TNT) * P[i];
        }

        return P;
    }

    private double[] construir_punto_q(Instance base){
        int i, j, cl, n;
        double TT, TNT, q;
        double[] baricentro , res, TNT_ls ;

        cl = (int) base.classValue();
        n = numValoresClase - 1;
        TT = getDistanceNearestNeighbors(base, cl);
        TNT = 0;
        TNT_ls = new double[numValoresClase-1];
        baricentro = getBaricentro();
        res = new double[n];

        if (qVersion == 2) {
            j = 0;
            for (i = 0; i < numValoresClase ; i++){
                if (i != cl) {
                    TNT_ls[j] = getDistanceNearestNeighbors(base, i);
                    j++;
                }
            }
            TNT = funcionResumen(TNT_ls);

            q = (wtt *TT + wtnt * (1 - TNT)) / (wtt + wtnt);
        }else{
            q = TT;
        }

        if (q == 1.)
            q -= EPSILON;

        for (i = 0; i < n; i++)
            res[i] = (1 - q) * basePoints[cl][i] + q * baricentro[i];

        if (m_Debug)
            escribirLog("\tPunto Q = " + q + " --> " + XNNUtils.vector2string(res) + "\n");

        return res;

    }



    private double funcionResumen(double[] arr){
        switch (this.summaryFunction){
            case 1:
                return arithmeticMean(arr);
            case 2:
                return geometricMean(arr);
            case 3:
                return harmonicMean(arr);
            case 4:
                return max(arr);
            case 5:
                return min(arr);
            default:
                return arithmeticMean(arr);
        }
    }

    private double arithmeticMean(double[] arr){
        int i;
        double sum = 0.;
        for (i = 0; i < arr.length; i++)
            sum += arr[i];
        return sum / arr.length;
    }

    private double geometricMean(double[] arr){
        int i;
        double accum = 0.;
        for (i = 0; i < arr.length; i++)
            accum *= arr[i];
        return Math.pow(accum, 1./arr.length);
    }

    private double harmonicMean(double[] arr){
        int i;
        double accum = 0.;
        for (i = 0; i < arr.length; i++)
            accum += 1/arr[i];
        return arr.length / accum;
    }

    private double max(double[] arr){
        int i;
        double aux = -1. ;
        for (i = 0; i < arr.length; i++) {
            if (arr[i] > aux)
                aux = arr[i];
        }
        return aux;
    }

    private double min(double[] arr){
        int i;
        double aux = Double.MAX_VALUE ;
        for (i = 0; i < arr.length; i++)
            if (arr[i] < aux)
                aux = arr[i];
        return aux;
    }

    private double getDistanceNearestNeighbors(Instance base, int clase){
        Instance g;
        int i, clase_g, j;
        int k = ((int)base.classValue()) == clase ? k1 : k2;
        double dist, max_dist;
        double[] dist_arr = new double[k];
        Enumeration enumInst = instances.enumerateInstances();

        for (i = 0; i < k; i++)
            dist_arr[i] = 1.0;

        while (enumInst.hasMoreElements()){
            g = (Instance) enumInst.nextElement();
            if (g.equals(base))
                continue;
            clase_g = (int) g.classValue();
            if (clase_g == clase){
                dist = distancia(g, base);
                j = 0;
                max_dist = 1.1;
                for (i = 0; i < k; i++){
                    if (max_dist < dist_arr[i]){
                        j=i;
                        max_dist = dist_arr[i];
                    }
                }
                if (dist < max_dist)
                    dist_arr[j] = dist;
            }
        }

        return funcionResumen(dist_arr);
    }

    /**
     * Calculates the distance between two instances
     *
     * @param first  the first instance
     * @param second the second instance
     * @return the distance between the two given instances, between 0 and 1
     */
    private double distancia(Instance first, Instance second) {

        double distance = 0;
        double diff;
        int n = first.numAttributes();
        double p = Math.pow(2, minkowskiDegree);

        for (int i = 0; i < n; i++) {
            if (i != instances.classIndex()) {
                diff = difference(i, first.value(i), second.value(i));
                distance += Math.pow(Math.abs(diff), p);
            }
        }

        return Math.pow(distance, 1./p);
    }

    private double difference(int index, double val1, double val2) {

        switch (instances.attribute(index).type()) {
            case Attribute.NOMINAL:
                if (Utils.isMissingValue(val1) || Utils.isMissingValue(val2) || (val1 != val2)) {
                    return 1;
                } else {
                    return 0;
                }
        /*
           // Distancia inventada para artibutos discretos. Gualberto.
           int n = training.numAttributes();
           int n2 = (int)Math.ceil(n / 2.0);
           double u = 0.01, m = 0.005, b = 0.99;

           index++; // Tratamiento de 1..n
           if (index <= n2)
               return ( b + u * index / n2 );
           else // n2 < index <= n
               return ( b + u * (n - index + ( (1 + Math.pow(-1.0, n + 1.0)) / 2 ) / n2) + m );
         } else {
            return 0;
        */
            case Attribute.NUMERIC:

                // If attribute is numeric
                if (Utils.isMissingValue(val1) ||
                        Utils.isMissingValue(val2)) {
                    if (Utils.isMissingValue(val1) &&
                            Utils.isMissingValue(val2)) {
                        return 1;
                    } else {
                        double diff;
                        if (Utils.isMissingValue(val2)) {
                            diff = val1;
                        } else {
                            diff = val2;
                        }
                        if (diff < 0.5) {
                            diff = 1.0 - diff;
                        }
                        return diff;
                    }
                } else {
                    return val1 - val2;
                }
            default:
                return 0;
        }
    }

    @Override
    public Enumeration<Option> listOptions() {
        Vector result = new Vector(20);

        result.addElement(new Option(
                "\tChoose the number of neighbors for the same\n"
                        + "\tclass\n" + "\t(Default = 3)",
                "K", 1, "-K <k1>"));
        result.addElement(new Option(
                "\tChoose the number of neighbors for other\n"
                        + "\tclasses\n" + "\t(Default = 3)",
                "k", 1, "-k <k2>"));
        result.addElement(new Option(
                "\tChoose among the different summary functions.\n"
                        + "\t(Default = 1)",
                "S", 1, "-S <summary function number>"));
        result.addElement(new Option(
                "\tChoose the Minkowski degree for distance calculation.\n"
                        + "\t(Default = 2.0)",
                "D", 1, "-D <Minkowski degree>"));
        result.addElement(new Option(
                "\tChoose the type of Q-point calculation.\n"
                        + "\t(Default = 1)",
                "Q", 1, "-Q <calculation of Q-point>"));
        result.addElement(new Option(
                "\tChoose the weight for distances to the same class used to create the Q-point\n"
                        + "\t(Default = 1)",
                "W", 1, "-W <weight for same class>"));
        result.addElement(new Option(
                "\tChoose the weight for distances to other classes used to create the Q point\n"
                        + "\t(Default = 1)",
                "w", 1, "-w <weight for different classes>"));

        result.addAll(Collections.list(super.listOptions()));

        return result.elements();
    }

    @Override
    public void setOptions(String[] options) throws Exception {

        String k1String = Utils.getOption('K', options);
        if (k1String.length() != 0) {
            setK1(Integer.parseInt(k1String));
        } else {
            setK1(3);
        }

        String k2String = Utils.getOption('k', options);
        if (k2String.length() != 0) {
            setK2(Integer.parseInt(k2String));
        } else {
            setK2(3);
        }

        String summFuncString = Utils.getOption('S', options);
        if (summFuncString.length() != 0) {
            setSummaryFunction(new SelectedTag((int)Double.parseDouble(summFuncString), GeometricEnhancementModel.SUMMARY_FUNC));
        } else {
            setSummaryFunction(new SelectedTag(1, GeometricEnhancementModel.SUMMARY_FUNC));
        }

        String minkownski_d_str = Utils.getOption('D', options);
        if(minkownski_d_str.length() != 0)
            setMinkowskiDegree(Double.parseDouble(minkownski_d_str));
        else
            setMinkowskiDegree(2.0);

        String qVersionString = Utils.getOption('Q', options);
        if (qVersionString.length() != 0) {
            setQVersion(new SelectedTag((int)Double.parseDouble(qVersionString), GeometricEnhancementModel.Q_VERSION));
        } else {
            setQVersion(new SelectedTag(1, GeometricEnhancementModel.Q_VERSION));
        }

        String wTTString = Utils.getOption('W', options);
        if (wTTString.length() != 0) {
            setWtt((int)Double.parseDouble(wTTString));
        } else {
            setWtt(1.0);
        }

        String wTNTString = Utils.getOption('w', options);
        if (wTNTString.length() != 0) {
            setWtnt((int)Double.parseDouble(wTNTString));
        } else {
            setWtnt(1.0);
        }

        super.setOptions(options);
        Utils.checkForRemainingOptions(options);
    }

    @Override
    public String[] getOptions() {
        Vector<String> result = new Vector<String>();

        result.add("-K");
        result.add("" + this.k1);
        result.add("-k");
        result.add("" + this.k2);
        result.add("-S");
        result.add("" + this.summaryFunction);
        result.add("-D");
        result.add("" + this.minkowskiDegree);
        result.add("-Q");
        result.add("" + this.qVersion);
        result.add("-W");
        result.add("" + this.wtt);
        result.add("-w");
        result.add("" + this.wtnt);

        Collections.addAll(result, super.getOptions());

        return result.toArray(new String[result.size()]);
    }

    public String QVersionTipText() {
        return "Tooltip";
    }

    public String k1TipText() {
        return "Tooltip";
    }

    public String k2TipText() {
        return "Tooltip";
    }

    public String minkowskiDegreeTipText() {
        return "Tooltip";
    }

    public String summaryFunctionTipText() {
        return "Tooltip";
    }

    public String wtntTipText() {
        return "Tooltip";
    }

    public String wttTipText() {
        return "Tooltip";
    }

    public String globalInfo() {
        return "XNN Geometric Enhancement Model 2 :\n\n(write explanation)\n\n";
    }

}
