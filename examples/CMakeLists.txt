
add_executable(fashion_mnist fashion_mnist.cpp)
target_link_libraries(fashion_mnist PRIVATE DeepModel)

add_executable(mnist mnist.cpp)
target_link_libraries(mnist PRIVATE DeepModel)

add_executable(linear_algebra linear_algebra.cpp)
target_link_libraries(linear_algebra PRIVATE DeepModel)