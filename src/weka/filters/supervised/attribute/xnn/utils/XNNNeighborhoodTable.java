package weka.filters.supervised.attribute.xnn.utils;

import weka.core.Instance;
import weka.core.Instances;

import java.text.DecimalFormat;
import java.util.*;

public class XNNNeighborhoodTable implements Iterable<XNNNeighborhood> {

    protected List<XNNNeighborhood> table;
    protected int knn;
    protected double minDistance, minDistanceRange, maxDistanceRange;


    public XNNNeighborhoodTable () {

        this(10);

    }

    public XNNNeighborhoodTable (int knn) {

        table = new LinkedList<XNNNeighborhood>();
        this.knn = knn;

    }

    public void build (Instances instances) {

        Enumeration<Instance> e = instances.enumerateInstances();
        XNNNeighborhood row;
        Instance inst; int i = 0;

        // For each instance, build its XNNNeighborhood object and add it to the table.
        while (e.hasMoreElements()) {
            inst = e.nextElement();
            row = new XNNNeighborhood(inst, i, knn);
            row.processNeighbors(instances);
            //System.out.println(row);
            table.add(row);
            i++;
        }
        postProcess();

    }

    protected void postProcess () {

        double x;

        // Assess minDistance, minDistanceRange and maxDistanceRange.
        minDistance = Double.MAX_VALUE;
        minDistanceRange = Double.MAX_VALUE;
        maxDistanceRange = Double.MIN_VALUE;
        for (XNNNeighborhood row : table) {
            x = row.getDistanceRange();
            if (x < minDistanceRange)
                minDistanceRange = x;
            if (x > maxDistanceRange)
                maxDistanceRange = x;
            x = row.getMinDistance();
            if (x < minDistance)
                minDistance = x;
        }

        // Update the relative density of all rows (XNNNeighborhood) of the table.
        x = maxDistanceRange - minDistanceRange;
        for (XNNNeighborhood row : table) {
            row.setRelativeDensity((row.getDistanceRange() - minDistanceRange) / x);
        }

    }

    public double getMinDistance() {
        return minDistance;
    }

    public double getMinDistanceRange() {
        return minDistanceRange;
    }

    public double getMaxDistanceRange() {
        return maxDistanceRange;
    }

    public void add (XNNNeighborhood row) {
        table.add(row);
    }

    @Override
    public Iterator<XNNNeighborhood> iterator() {
        return table.iterator();
    }

    @Override
    public String toString() {
        return "XNNNeighborhoodTable{" +
                "minDistance=" + new DecimalFormat("#0.000").format(minDistance) +
                ", minDistanceRange=" + new DecimalFormat("#0.000").format(minDistanceRange) +
                ", maxDistanceRange=" + new DecimalFormat("#0.000").format(maxDistanceRange) +
                ", table=" + table +
                '}';
    }
}
