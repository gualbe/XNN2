package weka.filters.supervised.attribute.xnn;

import weka.core.OptionHandler;
import weka.core.Tag;
import weka.filters.supervised.attribute.xnn.utils.XNNUtils;

public abstract class GeometricEnhancementModel extends EnhancementModel implements OptionHandler {

    public static final Tag[] SUMMARY_FUNC = {
            new Tag(1, "Arithmetic mean"),
            new Tag(2, "Geometric mean"),
            new Tag(3, "Harmonic mean"),
            new Tag(4, "Maximum "),
            new Tag(5, "Minimum "),
    };

    /**
     * Puntos base del modelo continuo para las etiquetas nominales.-
     * <p>
     * Se trata de un conjunto de n puntos en R^(n-1) separados entre sí
     * a distancia unitaria (n = numValoresClase) necesarios para la
     * construcción de las tuplas de números reales asociadas a las
     * etiquetas nominales de las instancias. Estas tuplas extienden la
     * semántica de dichas etiquetas incorporando información sobre la
     * cercanía a las instancias que tienen etiquetas distintas (enemigos).
     */
    protected static double[][] basePoints;
    protected static int numBasePoints = 0;

    public static final Tag[] Q_VERSION = {
            new Tag(1, "Version 1"),
            new Tag(2, "Version 2")
    };





    public GeometricEnhancementModel() {
        super();
    }

    public GeometricEnhancementModel(boolean noNorm, boolean m_Debug, boolean toFileDebug){
        super(noNorm, m_Debug, toFileDebug);
    }

    public static double[][] getBasePoints(int numClasses) {
        if (numBasePoints != numClasses) {
            basePoints = buildBasePoints(numClasses);
            numBasePoints = numClasses;
        }
        return basePoints;
    }

    @Override
    protected void init(){

    }

    private static double [][] buildBasePoints(int numClasses) {
        int i, t, z;
        double [][] basePoints;

        // Leer del fichero de puntos el número de puntos (z) ya calculados y
        // almacenados.
        z = 0;  // De momento hasta que no trate con ficheros.

        basePoints = new double[numClasses][numClasses - 1];
        t = Math.min(numClasses, z);
        for (i = 1; i <= t; i++) {
            // leer(fich, puntos_base[i])
        }

        if (t < numClasses) {  // Hay que construir más puntos.
            for (i = t + 1; i <= numClasses; i++) {
                basePoints[i - 1] = buildBasePoint(i, numClasses - 1, basePoints);
            }
        }

        return basePoints;
    }

    private static double [] buildBasePoint(int j, int n, double [][] basePoints) {
        // Supuesto calculados los j-1 puntos anteriores y que 1<=j<=n

        int i, x;
        double t;
        double[] baricentro;
        double [] res = new double[n];

        if (j < 3) {
            if (j == 1)
                res[0] = 0;
            else
                res[0] = 1;
            for (i = 1; i < n; i++)
                res[i] = 0;
        } else {
            // Cálculo del baricentro
            baricentro = puntoMedio(basePoints, j - 1, j - 2, -1);

            // Cálculo de la intersección de la recta normal al conjunto de puntos
            // ya calculados que pasa por el baricentro con la esfera de radio
            // unitario centrada en cualquiera de los puntos ya calculados.
            // El resultado de este cálculo es la primera coordenada igual a 0
            // del baricentro.
            t = 1.0;
            for (x = 0; x < j - 2; x++) {
                t = t - Math.pow(baricentro[x], 2.0);
            }
            t = Math.sqrt(t);

            // Finalmente: Obtención del punto base j.
            for (x = 0; x < j - 2; x++) {
                res[x] = baricentro[x];
            }
            res[j - 2] = t;
            for (x = j - 1; x < n; x++) {
                res[x] = 0;
            }
        }

        return res;
    }

    protected  double[] getBaricentro(){
        double[] res = new double[numValoresClase - 1];
        double accum;
        int i, j;

        for(i = 0; i < numValoresClase - 1; i++){
            accum = 0.;
            for (j = 0; j < i+2; j++){
                accum += basePoints[j][i];
            }
            res[i] = accum / (i + 2);
        }

        return res;
    }


    /**
     * Calcula el punto medio o baricentro de un conjunto de puntos
     *
     * @param puntos Los puntos sobre los que trabajar
     * @param m      El número de puntos a considerar
     * @param n      El número de coordenadas para los puntos a considerar
     * @param salto  Si es mayor o gual que 0, ignora dicho punto en el cálculo
     */
    private static double[] puntoMedio(double[][] puntos, int m, int n, int salto) {
        int i, x;
        double[] res = new double[n];

        for (x = 0; x < n; x++) {
            for (i = 0; i < m; i++) {
                if (i != salto) {
                    res[x] += puntos[i][x];
                }
            }
            if (salto < 0)
                res[x] /= m;
            else
                res[x] /= m - 1;
        }
        return res;
    }


    protected void presentar_puntos_base() {
        int i, j;

        for (i = 0; i < numValoresClase; i++) {
            escribirLog("\tPunto " + i + " = ");
            escribirLog(XNNUtils.vector2string(basePoints[i]));
            escribirLog("\n");
        }
    }

}
