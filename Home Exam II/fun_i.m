function [ y_n ] = fun_i(w, x_n, sigma)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
y_n = x_n*w + normrnd(0,sigma,[size(x_n,1),1]); 
