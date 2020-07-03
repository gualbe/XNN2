package weka.filters.supervised.attribute.xnn;

import weka.core.*;
import weka.filters.supervised.attribute.xnn.utils.XNNUtils;

import java.util.*;

public class GeometricEnhancementModel1 extends GeometricEnhancementModel implements OptionHandler {



    public GeometricEnhancementModel1(boolean noNorm, boolean m_Debug, boolean toFileDebug, boolean preprocessDebug) {
        super(noNorm, m_Debug, toFileDebug);
    }

    @Override
    public Map<String, double[][]> generateEnhancementVectors() {
        int i, n, clase_g;
        double[] ned = new double[numValoresClase];
        Instance g;

        double[][] xnnpoints = new double[instances.numInstances()][numValoresClase-1];
        Map<String, double[][]> res = Collections.synchronizedMap(new LinkedHashMap<String, double[][]>());

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
            obtener_ned(g,ned);
            if (m_Debug)
                presentar_ned(ned, clase_g);
            xnnpoints[i] = extenderClaseNominal(clase_g, ned);
            if (m_Debug)
                escribirLog("}\n");
            i++;
        }

        res.put("XNNPoints", xnnpoints);

        return res;
    }

    @Override
    public Instances determineOutputFormat(Instances inputFormat) {
        Instances result = new Instances(inputFormat, 0);

        int n = inputFormat.numClasses();
        for (int i = 0; i < n - 1; i++) {
            result.insertAttributeAt(new Attribute("XNNPoint" + i), result.classIndex());
        }

        return result;
    }


    // METODOS DE CALCULO DE ARTEFACTOS


    private void obtener_ned(Instance base, double[] ned) {
        Instance g;
        double dist;
        int i, clase_base, clase_g;
        Enumeration enum1 = instances.enumerateInstances();

        // Inicialización de ned
        for (i = 0; i < numValoresClase; i++)
            ned[i] = 1.0;

        clase_base = (int) base.classValue();
        while (enum1.hasMoreElements()) {
            g = (Instance) enum1.nextElement();
            clase_g = (int) g.classValue();
            if (clase_g != clase_base) {
                dist = distancia(g, base);
                if (dist < ned[clase_g]) {
                    ned[clase_g] = dist;
                }
            }
        }
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

        for (int i = 0; i < n; i++) {
            if (i != instances.classIndex()) {
                diff = difference(i, first.value(i), second.value(i));
                distance += diff * diff;
            }
        }
        //TODO: Se no es distance/n ya que n también incluye la clase. debería ser distance/numAtributosNoClase. Mirar implementación en modelo geométrico 2.
        return Math.sqrt(distance / n);
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



    private double[] extenderClaseNominal(int cl, double[] ned) {
        // Pre: 0 <= cl <= numValoresClase-1 //

        double[] res = new double[numValoresClase - 1];
        double[][] pm = new double[numValoresClase][numValoresClase - 1];
        double[][] ph = new double[numValoresClase][numValoresClase - 1];
        double[] x = new double[numValoresClase];
        double nedT;
        int i, j;

      /*
        Paso 1:  Cálculo de los puntos medios (pm) de los segmentos que unen la
        ------   clase 'cl' con las demás clases enemigas.
      */
        for (i = 0; i < numValoresClase; i++) {
            if (i != cl) {
                pm[i] = calcularPM(basePoints[cl], basePoints[i]);
                if (m_Debug) {
                    escribirLog("\tPunto medio con la etiqueta " + i + " = ");
                    escribirLog(XNNUtils.vector2string(pm[i]));
                    escribirLog("\n");
                }
            }
        }

      /*
        Paso 2:  Cálculo de los puntos (ph) que relacionan o enfrentan la clase
        ------   'cl' con las demás clases.
      */
        for (i = 0; i < numValoresClase; i++) {
            if (i != cl) {
                ph[i] = calcularPH(basePoints[cl], pm[i], ned[i]);
                if (m_Debug) {
                    escribirLog("\tPunto H enfrentado con la etiqueta " + i + " {\n");
                    escribirLog("\t\tPunto Casa = ");
                    escribirLog(XNNUtils.vector2string(basePoints[cl]));
                    escribirLog("\n\t\tPunto Medio Enfrentado = ");
                    escribirLog(XNNUtils.vector2string(pm[i]));
                    escribirLog("\n\t\tned(" + i + ") = " + ned[i]);
                    escribirLog("\n\t\tPunto H = ");
                    escribirLog(XNNUtils.vector2string(ph[i]));
                    escribirLog("\n\t}\n");
                }
            }
        }

      /*
        Paso 3:  Cálculo de la suma total (nedT) de ned(i).
        ------
      */
        nedT = numValoresClase - 1;
        for (i = 0; i < numValoresClase; i++) {
            if (i != cl) {
                nedT = nedT - ned[i];
            }
        }
        if (m_Debug) {
            escribirLog("\tSuma total nedT = " + nedT + "\n");
        }


      /*
        Paso 4:  Cálculo de las contribuciones finales (x).
        ------
      */
        if (m_Debug)
            escribirLog("\tContribuciones finales x(i) {\n");
        for (i = 0; i < numValoresClase; i++) {
            if (i != cl) {
                if (nedT == 0)
                    x[i] = 0.5;
                else
                    x[i] = (1 - ned[i]) / nedT;
                if (m_Debug)
                    escribirLog("\t\tx(" + i + ") = " + x[i] + "\n");
            }
        }
        if (m_Debug)
            escribirLog("\t}\n");


      /*
        Paso 5:  Finalmente, cálculo del punto a devolver (res).
        ------
      */
        if (m_Debug)
            escribirLog("\tPunto final asignado a la instancia {\n");

        for (i = 0; i < numValoresClase - 1; i++) {
            // No hace falta res[i] = 0 porque Java ya inicializa a 0
            if (m_Debug)
                escribirLog("\t\tCoord " + i + " = sum[j, (1-x(j)) * H(j," + i + ")] = ");
            for (j = 0; j < numValoresClase; j++) {
                if (j != cl) {
                    res[i] += x[j] * ph[j][i];
                    if (m_Debug && j != numValoresClase - 1)
                        escribirLog(x[j] + " * " + ph[j][i] + " + ");
                }
            }
            //res[i] /= numValoresClase - 1;
            if (m_Debug)
                escribirLog(x[j - 1] + " * " + ph[j - 1][i] + " = " + res[i] + "\n");
        }

        // Para la variante del clasificador con puntos medios sin contribuciones
        //res = puntoMedio(ph, numValoresClase, numValoresClase-1, cl);

        if (m_Debug) {
            escribirLog("\t\tF = ");
            escribirLog(XNNUtils.vector2string(res));
            escribirLog("\n\t}\n");
        }

      /*
      // Comprobar integridad en la construcción del punto asociado a la instancia
      double [] d = new double[2];
      StringBuffer txt = new StringBuffer();
      d = dist_a_puntos_base(res, txt);
      if ( ((int)d[1]) == cl ) {
          // Construcción correcta
          if (m_Debug && preprocessDebug) {
              escribirLog("Ok\n");
          }
      }
      else {
          // Construcción incorrecta
          if (m_Debug && preprocessDebug) {
              escribirLog("---!-!-!-!--> MAL <--!-!-!-!---\n");
          }
      }
      */

        return res;
    }

    private double[] calcularPM(double[] p1, double[] p2) {
        int i, n = p1.length;
        double[] res = new double[n];

        for (i = 0; i < n; i++) {
            res[i] = (p1[i] + p2[i]) / 2.0;
        }

        return res;
    }

    private double[] calcularPH(double[] p1, double[] p2, double lambda) {
        // Pre: p1.length = p2.length  Y  0 <= lambda <= 1 //

        int n = p1.length;
        double[] res = new double[n];

        for (int i = 0; i < n; i++) {
            res[i] = lambda * p1[i] + (1 - lambda) * p2[i];
        }

        return res;
    }

    // METODOS DE DEBUG

    private void presentar_ned(double[] ned, int cl) {
        escribirLog("\tVector ned = [");
        for (int i = 0; i < numValoresClase - 1; i++) {
            if (i != cl)
                escribirLog(ned[i] + ", ");
            else
                escribirLog("-, ");
        }
        if (cl != numValoresClase - 1)
            escribirLog("" + ned[numValoresClase - 1]);
        else
            escribirLog("-");
        escribirLog("]\n");
    }


    @Override
    public Enumeration<Option> listOptions() {
        return null;
    }

    @Override
    public void setOptions(String[] options) throws Exception {

    }

    @Override
    public String[] getOptions() {
        return new String[0];
    }
}
