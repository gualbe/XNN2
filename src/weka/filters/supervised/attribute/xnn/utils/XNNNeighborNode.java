package weka.filters.supervised.attribute.xnn.utils;

import weka.core.Instance;

public class XNNNeighborNode {

    /**
     * The neighbor instance
     */
    public Instance m_Instance;

    /**
     * The distance from the current instance to this neighbor
     */
    public double m_Distance;

    /**
     * The index of the instance in the dataset
     */
    public int index;

    /**
     * A link to the next neighbor instance
     */
    public XNNNeighborNode m_Next;

    /**
     * Create a new neighbor node.
     *
     * @param distance the distance to the neighbor
     * @param instance the neighbor instance
     * @param next     the next neighbor node
     */
    public XNNNeighborNode(double distance, Instance instance, int index, XNNNeighborNode next) {
        m_Distance = distance;
        m_Instance = instance;
        this.index = index;
        m_Next = next;
    }

    /**
     * Create a new neighbor node that doesn't link to any other nodes.
     *
     * @param distance the distance to the neighbor
     * @param instance the neighbor instance
     */
    public XNNNeighborNode(double distance, Instance instance, int index) {

        this(distance, instance, index, null);
    }
}
