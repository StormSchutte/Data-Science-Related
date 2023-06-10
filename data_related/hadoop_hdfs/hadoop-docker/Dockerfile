# mkdir hadoop-docker
# cd hadoop-docker


# Use a base image with OpenJDK
FROM openjdk:8-jdk

# Set environment variables
ENV HADOOP_VERSION=3.3.0
ENV HADOOP_HOME=/opt/hadoop
ENV HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
ENV PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin

# Install required packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget

# Download and install Hadoop
RUN wget -q https://archive.apache.org/dist/hadoop/common/hadoop-$HADOOP_VERSION/hadoop-$HADOOP_VERSION.tar.gz && \
    tar -xzf hadoop-$HADOOP_VERSION.tar.gz -C /opt && \
    rm hadoop-$HADOOP_VERSION.tar.gz && \
    ln -s /opt/hadoop-$HADOOP_VERSION $HADOOP_HOME

# Copy configuration files
COPY core-site.xml hdfs-site.xml mapred-site.xml yarn-site.xml $HADOOP_CONF_DIR/

# Expose required ports
EXPOSE 9870 9864 9866 9867 19888 8088 8888 8030 8031 8032 8033 8040 8042 8080

# Start Hadoop services
CMD ["sh", "-c", "start-dfs.sh && start-yarn.sh && tail -f /dev/null"]
