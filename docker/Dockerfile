FROM nvidia/cuda

# Install software
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -y install openssh-server wget curl git htop screen tmux sudo

ADD ./setup/run.sh /run.sh
RUN chmod +x /*.sh

# SSH-Authentication
RUN mkdir -p /var/run/sshd && sed -i "s/UsePrivilegeSeparation.*/UsePrivilegeSeparation no/g" /etc/ssh/sshd_config \
  && sed -i 's/PermitRootLogin without-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
  && touch /root/.Xauthority \
  && true

# Set pub keys
RUN mkdir -p "root/.ssh"
RUN chmod go-rwx "root/.ssh"
COPY ./setup/ssh_keys.pub "root/.ssh/authorized_keys"
RUN chmod go-rw "root/.ssh/authorized_keys"

EXPOSE 22
CMD ["/run.sh"]