

## install

install MySQL server
```
sudo apt install mysql-server -y
sudo systemctl start mysql.service
```

```
sudo systemctl status mysql.service
```


## uninstall 
```
sudo apt purge mysql-server mysql-client mysql-common -y
sudo rm -rf /etc/mysql /var/lib/mysql
sudo apt autoremove -y
```

## secure mysql installation
```
sudo mysql_secure_installation
```
Follow the prompts to:

- Set a root password (important for security).
- Remove anonymous users.
- Disallow root login remotely.
- Remove the test database.
- Reload privileges.

log in mysql using root
```
sudo mysql -u root -p
```
ps= root


## configuration

CREATE USER 'admin'@'localhost' IDENTIFIED BY 'strong_password';
GRANT ALL PRIVILEGES ON *.* TO 'admin'@'localhost' WITH GRANT OPTION;
FLUSH PRIVILEGES;
EXIT;

mysql -u admin -p

if there is error:
ERROR 2002 (HY000): Can't connect to local MySQL server through socket '/tmp/mysql.sock' (2)

```
sudo nano /etc/mysql/my.cnf
```
Add 
```
[client]
socket = /var/run/mysqld/mysqld.sock

[mysqld]
socket = /var/run/mysqld/mysqld.sock



## add database
```
mysql -u admin -p
```
enter the password "strong_password". then enter MySQL. the next, 
create a database "IEDB"

mysql> DROP DATABASE IEDB;  
mysql> CREATE DATABASE IEDB;  
mysql> Exit  


Then import dump file into the database at another terminal
```
sudo mysql -u admin -p IEDB < /home/yuan/data/IEDB/iedb_public.sql
```

## set configuration for MYSQL connector
SET GLOBAL wait_timeout = 28800;
SET GLOBAL interactive_timeout = 28800;
SET GLOBAL max_allowed_packet = 256*1024*1024;  # 256MB