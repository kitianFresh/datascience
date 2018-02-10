
drop database if exists `StudentManager`;
create database `StudentManager`;
use `StudentManager`;

drop table if exists `students`;
create table `students` (
    `Id` int unsigned auto_increment not null,
    `Name` varchar(255) not null,
    `Class` int unsigned,
    `Age` int unsigned,
    `Sex` varchar(255),
    `Math` int,
    `Computer` int,
    `Physics` int,
    `Registry` timestamp not null,
    primary key (`Id`)
)

