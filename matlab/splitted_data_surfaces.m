az = readmatrix("matlab/data/azimuth.txt");
el = readmatrix("matlab/data/elevation.txt");


x1 = az(:,1);
y1 = az(:,2);
z1 = az(:,3);

x2 = el(:,1);
y2 = el(:,2);
z2 = el(:,3);

curveFitter(x1,y1,z1);
curveFitter(x2,y2,z2);
% curveFitter(x3,y3,z3);
% curveFitter(x4,y4,z4);
% plot3(x2,y2,z2,'-o','Color','b','MarkerSize',10,...
%     'MarkerFaceColor','#D9FFFF');


