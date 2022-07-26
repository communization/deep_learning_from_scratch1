#완전연결 계층(Affine 계층)의 특징
#인접하는 계층의 뉴런이 모두 연결되고 출력의 수는 임의로 정할 수 있다.
#Affine 계층의 문제점
#데이터의 형상이 무시된다. (색상을 포함한 데이터는 3차원 이지만, Affine은 1차원 데이터로 평탄화 해줘야함.)
#그러나, 합성곱 계층은 형상을 유지하고, 3차원 데이터로 전달한다.
#CNN에서는 합성곱 계층의 입출력 데이터를 특징맵feature map이라고도 한다.

#필터를 적용하는 위치의 간격을 스트라이드라고 한다. 스트라이드를 2로하면 윈도우가 두칸씩 이동한다.
#출력 크기는 p.234에 식으로 나와 있다. (정수로 나누어 떨어지지 않는다면 가장 가까운 정수로 반올림하는 등의 방법 이용)

#길이방향(채널방향)으로 길이가 늘어난 3차원 합성곱을 진행한다면, 주의할점은
#입력데이터의 채널 수와 필터의 채널 수가 같아야 한다는 것이다.

#입력데이터와 필터는 채널수가 같아야 하고, 이는 한 층의 출력데이터를 얻는다.
#n개의 필터를 적용한다면 nc으의 출력데이터를 얻게 되는 것이다.