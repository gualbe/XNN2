package weka.filters.supervised.attribute.xnn.utils;

import weka.core.Instance;
import weka.core.Instances;

import java.text.DecimalFormat;
import java.util.*;

public class XNNNeighborhood implements Iterable<XNNNeighbor> {

    protected Instance baseInstance;
    protected int index, knn;
    protected double baseClass;
    protected Queue<XNNNeighbor> neighbors;
    protected double distanceRange;
    protected double minDistance, maxDistance;
    protected double relativeDensity;



    public XNNNeighborhood(Instance baseInstance, int index, int knn) {
        this.baseInstance = baseInstance;
        this.index = index;
        this.knn = knn;
        baseClass = baseInstance.classValue();
        neighbors = new PriorityQueue<XNNNeighbor>(knn); // Uso ProrityQueue pues necesito acceso rapido, O(cte) ,al vecino de mayor distancia
        distanceRange = 0;
        minDistance = 0;
        maxDistance = 0;
        relativeDensity = 0;
    }

    public void processNeighbors (Instances instances) {

        Enumeration<Instance> e = instances.enumerateInstances();
        Instance instance; int i = 0;

        // For each instance, compute distance to the base instance
        // and eventually add it to the neighbors collection.
        while (e.hasMoreElements()) {
            instance = e.nextElement();
            if (instance != baseInstance)
                processNeighbor(instance, i);
            i++;
        }

        // Compute the distance range as the difference between the maximum (peek()) and
        // minimum (max(), due to the reverse order of compareTo) distances on neighbors.
        this.minDistance = Collections.max(neighbors).distance;
        this.maxDistance = neighbors.peek().distance;
        this.distanceRange = this.maxDistance - this.minDistance;

    }

    protected void processNeighbor (Instance neighbor, int index) {

        double dist;

        // Step 1. Compute the distance between the base instance and the neighbor.
        dist = XNNUtils.distance(baseInstance, neighbor);

        // Step 2. Add the neighbor to the collection, if proceed.
        if (neighbors.size() < knn)
            neighbors.offer(new XNNNeighbor(index, neighbor.classValue(), dist));
        else if (dist < neighbors.peek().distance) {
            neighbors.poll();
            neighbors.offer(new XNNNeighbor(index, neighbor.classValue(), dist));
        }
    }

    public Instance getBaseInstance() {
        return baseInstance;
    }

    public int getIndex() {
        return index;
    }

    public void setIndex(int index) {
        this.index = index;
    }

    public double getBaseClass() {
        return baseClass;
    }

    public void setBaseClass(double baseClass) {
        this.baseClass = baseClass;
    }

    public double getDistanceRange() {
        return distanceRange;
    }

    public double getRelativeDensity() {
        return relativeDensity;
    }

    public int getKnn() {
        return knn;
    }

    public double getMinDistance() {
        return minDistance;
    }

    public double getMaxDistance() {
        return maxDistance;
    }

    public void setRelativeDensity(double relativeDensity) {
        this.relativeDensity = relativeDensity;
    }

    public int getNumValoresClase () {
        return baseInstance.numClasses();
    }

    @Override
    public String toString() {
        return "\nXNNNeighborhood{" +
                "index=" + index +
                ", baseClass=" + baseClass +
                ", distanceRange=" + new DecimalFormat("#0.000").format(distanceRange) +
                ", minDistance=" + new DecimalFormat("#0.000").format(minDistance) +
                ", maxDistance=" + new DecimalFormat("#0.000").format(maxDistance) +
                ", relativeDensity=" + new DecimalFormat("#0.000").format(relativeDensity) +
                ", neighbors=" + neighbors +
                '}';
    }

    @Override
    public Iterator<XNNNeighbor> iterator() {
        return this.neighbors.iterator();
    }

}
